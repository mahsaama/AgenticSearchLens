"""invitro replay: re-sends invivo user prompts to each platform's own API
(with Web search on/off/forced) so responses can be compared against what
the platform originally returned.

Supports OpenAI (ChatGPT/GPT), xAI (Grok) via the OpenAI-compatible
Responses API, and Anthropic (Claude) / DeepSeek via the Anthropic-compatible
Messages API. Requires the relevant provider API key in .env -- see
README.md's "Setup" section.

Run directly (`python -m src.replays.chat_replayer`) to replay the models
configured in the __main__ block below; import `replayer()` to drive it
programmatically.
"""

import ast
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.common_io import OUTPUT_PATH
from src.web_search_decision.chatgpt_extraction import load_whole_data_from_file, load_web_data_from_file

load_dotenv()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
SKIPPED_REPLAY_SAMPLE_INDICES = {}

ANNOTATIONS_TURNS_PATH = (
    f"{OUTPUT_PATH}/metadata/Annotations_Turns_all.csv"
)
ANNOTATION_REQUIRED_COLUMNS = {
    "conv_id",
    "personal_presence",
    "special_category_presence",
}


model_replacements = {
    "gpt-5-1": "gpt-5.1-2025-11-13",
}

PROVIDER_ALIASES = {
    "openai": "openai",
    "chatgpt": "openai",
    "claude": "claude",
    "anthropic": "claude",
    "grok": "grok",
    "xai": "grok",
    "deepseek": "deepseek",
}

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "grok"}
ANTHROPIC_COMPATIBLE_PROVIDERS = {"claude", "deepseek"}
DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-5.3-chat-latest",
    "claude": "claude-sonnet-4-6",
    "grok": "grok-4.3",
    "deepseek": "deepseek-v4-flash",
}
tool_choices = ["auto"]
SYSTEM_PROMPT_DIR = Path(f"{OUTPUT_PATH}/replays/system_prompts")


def _import_openai():
    """Lazily import the openai package's client class (avoids requiring it
    for callers that only replay Anthropic-compatible providers)."""
    from openai import OpenAI

    return OpenAI


def _import_anthropic():
    """Lazily import the anthropic package's client class (avoids requiring
    it for callers that only replay OpenAI-compatible providers)."""
    from anthropic import Anthropic

    return Anthropic


def _normalize_provider(provider):
    """Map a provider/platform name (e.g. "chatgpt", "anthropic", "xai") to
    its canonical id: "openai", "claude", "grok", or "deepseek"."""
    key = (provider or "openai").strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def _resolve_developer_prompt(developer_prompt):
    """Resolve a developer-prompt argument to literal prompt text.

    A string ending in ".md" is treated as a filename to read from
    (relative to cwd, then SYSTEM_PROMPT_DIR); anything else is used
    verbatim as the prompt text itself."""
    if not isinstance(developer_prompt, str):
        return None

    candidate = developer_prompt.strip()
    if not candidate:
        return None

    possible_paths = []
    if candidate.endswith(".md"):
        possible_paths.append(Path(candidate))
        possible_paths.append(SYSTEM_PROMPT_DIR / candidate)

    for path in possible_paths:
        if path.exists() and path.is_file():
            return path.read_text().strip()

    return candidate


def _infer_provider_from_model(model):
    """Guess the provider from a model name prefix when none is given
    explicitly (e.g. "claude-sonnet-4-6" -> "claude"), defaulting to
    "openai"."""
    model_key = (model or "").strip().lower()
    if model_key.startswith("claude"):
        return "claude"
    if model_key.startswith("grok"):
        return "grok"
    if model_key.startswith("deepseek"):
        return "deepseek"
    return "openai"


def _openai_compatible_base_url(provider):
    """Optional API base-URL override for an OpenAI-compatible provider,
    from OPENAI_BASE_URL / XAI_BASE_URL in .env, or None to use the SDK's
    own default endpoint."""
    env_base_urls = {
        "openai": os.getenv("OPENAI_BASE_URL", "").strip(),
        "grok": os.getenv("XAI_BASE_URL", "").strip(),
    }
    return env_base_urls.get(provider) or None


def _client_for_provider(provider):
    """Construct an authenticated OpenAI or Anthropic SDK client for the
    given provider, reading its API key (and optional base URL) from .env."""
    provider = _normalize_provider(provider)
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        OpenAI = _import_openai()
        api_key_env = {
            "openai": "OPENAI_API_KEY",
            "grok": "XAI_API_KEY",
        }
        kwargs = {"api_key": os.getenv(api_key_env[provider])}
        base_url = _openai_compatible_base_url(provider)
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    if provider in ANTHROPIC_COMPATIBLE_PROVIDERS:
        Anthropic = _import_anthropic()
        api_key_env = {
            "claude": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        base_url_env = {
            "claude": "ANTHROPIC_BASE_URL",
            "deepseek": "DEEPSEEK_ANTHROPIC_BASE_URL",
        }
        kwargs = {"api_key": os.getenv(api_key_env[provider])}
        base_url = os.getenv(base_url_env[provider], "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        return Anthropic(**kwargs)

    raise ValueError(f"unsupported replay provider: {provider}")


def _as_list(value):
    """Coerce a value that may be a real list, a numpy array, a
    stringified-list (as stored in a CSV/pickle column), NaN, or a scalar
    into an actual Python list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return [value]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    return [value]


def _history_turn_depth(history_depth):
    """Convert a message-count history_depth into a turn count (2 messages
    per turn), or None if history_depth is None (meaning "no limit")."""
    if history_depth is None:
        return None
    return max(history_depth // 2, 0)


def _clean_messages(messages):
    """Coerce `messages` to a list (see _as_list) and drop blank entries."""
    return [str(msg).strip() for msg in _as_list(messages) if str(msg).strip()]


def _has_exact_history_depth(row, prior_turns):
    """True if `row` has exactly one more user message than `prior_turns`
    (i.e. prior_turns of history plus the current prompt) and at least
    `prior_turns` assistant replies -- used to select rows with a known,
    fixed amount of conversational history to replay with."""
    user_prompts = _clean_messages(row["user_msg_history"])
    assistant_prompts = _clean_messages(row["assistant_msg_history"])
    return (
        len(user_prompts) == prior_turns + 1
        and len(assistant_prompts) >= prior_turns
    )


def _safe_annotation_conv_ids(annotation_path=ANNOTATIONS_TURNS_PATH):
    """Return the set of conv_ids whose turns were all annotated as free of
    personal/special-category information (the PII-sanitization pass
    described in README.md's invitro setup), from a PII-annotation CSV --
    an internal research artifact, not shipped with this repo. Raises if
    `annotation_path` doesn't have the expected columns; the caller is
    expected to point this at their own equivalent annotation file before
    calling filter_df_for_history()/replayer().

    Deliberately not made "topic-independent" the way missing topic
    annotations are (see topic_classifier.py): this file exists specifically
    to keep personal/special-category conversations out of what gets sent
    to external provider APIs during replay, so a missing annotation file
    fails loudly here rather than silently treating everything as safe.
    """
    try:
        annotations = pd.read_csv(
            annotation_path,
            usecols=lambda column: column in ANNOTATION_REQUIRED_COLUMNS,
            dtype=str,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"PII-safety annotation file not found: {annotation_path}. "
            "invitro replay refuses to run without it, since it's what keeps "
            "personal/special-category conversations out of what gets sent "
            "to external provider APIs. Annotate your own data with the "
            "'conv_id', 'personal_presence', 'special_category_presence' "
            "columns (see Appendix C.1 of the paper) at this path, or pass "
            "annotation_path= to point elsewhere."
        ) from exc
    missing_columns = ANNOTATION_REQUIRED_COLUMNS.difference(annotations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"annotation file is missing required columns: {missing}")

    annotations = annotations.dropna(subset=["conv_id"])
    is_safe = (
        annotations["personal_presence"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("no")
        & annotations["special_category_presence"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("no")
    )
    safe_by_conv = is_safe.groupby(annotations["conv_id"].astype(str)).all()
    return set(safe_by_conv[safe_by_conv].index)


def _filter_to_safe_conversations(df, safe_conv_ids):
    """Drop every row whose conv_id isn't in `safe_conv_ids` (see
    _safe_annotation_conv_ids)."""
    return df[df["conv_id"].astype(str).isin(safe_conv_ids)].copy()


def filter_df_for_history(
    history_depth=0,
    samples_per_source=1,
    random_seed=RANDOM_SEED,
):
    """Build the sample of rows to replay: PII-safe, English-language turns
    with exactly `history_depth` messages of prior context, split evenly
    between turns that originally invoked Web search ("web") and ones that
    didn't ("non_web"), up to `samples_per_source` of each. Returns a
    DataFrame with a "sample_source" column recording which group each row
    came from.
    """
    whole_df = load_whole_data_from_file(fmt="pkl")
    web_df = load_web_data_from_file(fmt="pkl")

    safe_conv_ids = _safe_annotation_conv_ids()
    whole_df = _filter_to_safe_conversations(whole_df, safe_conv_ids)
    web_df = _filter_to_safe_conversations(web_df, safe_conv_ids)

    prior_turns = _history_turn_depth(history_depth)
    if prior_turns is None:
        prior_turns = 1

    def _starter_filter(df):
        df = df.copy()
        df["user_msg_history"] = df["user_msg_history"].apply(_as_list)
        df["assistant_msg_history"] = df["assistant_msg_history"].apply(_as_list)
        history_mask = df.apply(
            lambda row: _has_exact_history_depth(row, prior_turns),
            axis=1,
        )
        return df[history_mask & (df["language"] == "en")].copy()

    web_filtered = _starter_filter(web_df)
    web_keys = set(zip(web_filtered["conv_id"], web_filtered["turn_id"]))

    whole_filtered = _starter_filter(whole_df)
    whole_non_web = whole_filtered[
        ~whole_filtered.apply(
            lambda row: (row["conv_id"], row["turn_id"]) in web_keys, axis=1
        )
    ].copy()

    # print(len(web_filtered))
    # print(len(whole_non_web))

    def _sample(df):
        if samples_per_source is None:
            return df.copy()
        return df.sample(
            min(samples_per_source, len(df)),
            random_state=random_seed,
        ).copy()

    web_sample = _sample(web_filtered)
    web_sample["sample_source"] = "web"

    whole_sample = _sample(whole_non_web)
    whole_sample["sample_source"] = "non_web"

    df = (
        pd.concat([web_sample, whole_sample], ignore_index=True)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )
    return df[
        df.apply(lambda row: _has_exact_history_depth(row, prior_turns), axis=1)
    ].copy()


def _tool_choices_for_provider(provider):
    """The Web-search tool_choice modes to replay a prompt under for
    `provider`. Currently the same fixed list (module-level `tool_choices`)
    for every provider; kept as its own function as the extension point if
    that ever needs to vary per provider."""
    return tool_choices


def _openai_web_kwargs(
    replay_model,
    prompt,
    tool_choice,
    developer_prompt=None,
):
    """Build the kwargs for an OpenAI/Grok Responses-API call: `tool_choice`
    is "auto"/"required"/"none", controlling whether/how the web_search tool
    is offered."""
    kwargs = {
        "model": replay_model,
        "input": prompt,
        "store": False,
    }
    if isinstance(developer_prompt, str) and developer_prompt.strip():
        kwargs["instructions"] = developer_prompt.strip()
    if tool_choice != "none":
        kwargs["include"] = ["web_search_call.action.sources"]
        kwargs["tools"] = [{"type": "web_search"}]
        kwargs["tool_choice"] = tool_choice
    return kwargs


def _anthropic_content_text(content_blocks):
    """Concatenate the text blocks of an Anthropic response's content list
    (skipping tool_use/thinking/etc. blocks) into one string."""
    texts = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            texts.append(getattr(block, "text", ""))
    return "\n".join(text for text in texts if text).strip()


def _anthropic_web_tools(tool_choice):
    """The `tools` payload offering Anthropic's web_search tool, or None to
    omit it entirely (tool_choice == "none")."""
    if tool_choice == "none":
        return None
    return [
        {
            "type": "web_search_20250305",
            "name": "web_search",
        }
    ]


def _anthropic_tool_choice(tool_choice):
    """The `tool_choice` payload forcing web_search to be called, or None to
    leave the choice to the model (tool_choice != "required")."""
    if tool_choice == "required":
        return {"type": "tool", "name": "web_search"}
    return None


def _create_openai_compatible_payload(
    client,
    provider,
    replay_model,
    prompt,
    tool_choice,
    developer_prompt=None,
):
    """Call the OpenAI/Grok Responses API and normalize the result to
    {"output_text": str, "response": dict}."""
    if provider in {"openai", "grok"}:
        kwargs = _openai_web_kwargs(
            replay_model,
            prompt,
            tool_choice,
            developer_prompt=developer_prompt,
        )
        response = client.responses.create(**kwargs)
        response_dump = response.model_dump()
        return {
            "output_text": response.output_text,
            "response": response_dump,
        }

    raise ValueError(f"unsupported OpenAI-compatible provider: {provider}")


def _anthropic_request_kwargs(
    replay_model,
    prompt,
    tool_choice,
    provider,
    developer_prompt=None,
):
    """Build the kwargs for an Anthropic Messages-API call (Claude or
    DeepSeek, via its Anthropic-compatible endpoint)."""
    kwargs = {
        "model": replay_model,
        "messages": prompt,
        "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS")),
    }
    if isinstance(developer_prompt, str) and developer_prompt.strip():
        kwargs["system"] = developer_prompt.strip()
    if _normalize_provider(provider) == "deepseek":
        kwargs["thinking"] = {"type": "disabled"}
    tools = _anthropic_web_tools(tool_choice)
    if tools is not None:
        kwargs["tools"] = tools
    forced_tool_choice = _anthropic_tool_choice(tool_choice)
    if forced_tool_choice is not None:
        kwargs["tool_choice"] = forced_tool_choice
    return kwargs


def _create_anthropic_payload(
    client,
    replay_model,
    prompt,
    tool_choice,
    provider,
    developer_prompt=None,
):
    """Call the Anthropic Messages API and normalize the result to
    {"output_text": str, "response": dict}, resuming the request if the
    model pauses mid-turn (stop_reason "pause_turn", seen during long
    web_search tool-use chains) until it finishes."""
    kwargs = _anthropic_request_kwargs(
        replay_model,
        prompt,
        tool_choice,
        provider,
        developer_prompt=developer_prompt,
    )
    response = client.messages.create(**kwargs)

    while getattr(response, "stop_reason", None) == "pause_turn":
        continuation_messages = list(prompt) + [
            {"role": "assistant", "content": response.content}
        ]
        continuation_kwargs = _anthropic_request_kwargs(
            replay_model,
            continuation_messages,
            tool_choice,
            provider,
            developer_prompt=developer_prompt,
        )
        response = client.messages.create(**continuation_kwargs)

    response_dump = response.model_dump()
    return {
        "output_text": _anthropic_content_text(response.content),
        "response": response_dump,
    }


def _create_response_payload(
    replay_model,
    prompt,
    tool_choice,
    replay_provider="openai",
    developer_prompt=None,
):
    """Send one replay request to `replay_provider`'s API and return
    {"output_text": str, "response": dict} -- the single entry point
    replayer() uses regardless of which provider it's replaying."""
    provider = _normalize_provider(replay_provider)
    client = _client_for_provider(provider)
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return _create_openai_compatible_payload(
            client,
            provider,
            replay_model,
            prompt,
            tool_choice,
            developer_prompt=developer_prompt,
        )
    if provider in ANTHROPIC_COMPATIBLE_PROVIDERS:
        return _create_anthropic_payload(
            client,
            replay_model,
            prompt,
            tool_choice,
            provider,
            developer_prompt=developer_prompt,
        )
    raise ValueError(f"unsupported replay provider: {provider}")


def _most_frequent_model(openai_models):
    """The most common non-empty model name in a turn's per-message model
    list (a turn can span several assistant messages/models), or None if
    there isn't one. Used to pick the "invivo_model" recorded alongside
    each replay result."""
    valid_models = [
        m
        for m in openai_models
        if isinstance(m, str) and m.strip() and m.lower() != "none"
    ]
    if not valid_models:
        return None
    return Counter(valid_models).most_common(1)[0][0]


def _save_replay_results(results, output_file):
    """Write the {result_key: result} replay-results dict to `output_file`
    as pretty-printed JSON."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def _load_replay_results(output_file):
    """Load a previously-saved replay-results dict, or {} if `output_file`
    is falsy/missing/not a dict -- lets replayer() resume an interrupted run
    without redoing already-completed requests."""
    if not output_file:
        return {}
    path = Path(output_file)
    if not path.exists() or not path.is_file():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _replay_output_file(model, dev_prompt=""):
    """Default results path for a given replay model (and, if replaying
    under another model's developer prompt, that prompt's name)."""
    if dev_prompt:
        return f"{OUTPUT_PATH}/replays/{model}_dev_prompt_{dev_prompt}.json"
    return f"{OUTPUT_PATH}/replays/{model}.json"


def _build_prompt(user_prompts, assistant_prompts, with_history, history_depth):
    """Build the message list to send for replay: the current (most recent)
    user prompt, optionally preceded by up to `history_depth` turns of
    prior user/assistant messages if `with_history` is set. Returns
    (prompt_messages, current_user_prompt_text)."""
    user_prompts = _clean_messages(user_prompts)
    assistant_prompts = _clean_messages(assistant_prompts)
    if not user_prompts:
        raise ValueError("row has no user messages to replay")

    current_user_idx = len(user_prompts) - 1
    current_user_prompt = user_prompts[current_user_idx]
    prompt = []

    if with_history:
        paired_history = list(zip(user_prompts, assistant_prompts))[:current_user_idx]
        max_prior_turns = len(paired_history)
        history_turns = _history_turn_depth(history_depth)
        if history_turns is None:
            history_turns = max_prior_turns
        history_turns = min(history_turns, max_prior_turns)

        selected_history = paired_history[-history_turns:] if history_turns else []
        for user_prompt, assistant_prompt in selected_history:
            prompt.append({"role": "user", "content": user_prompt})
            prompt.append({"role": "assistant", "content": assistant_prompt})

    prompt.append({"role": "user", "content": current_user_prompt})
    return prompt, current_user_prompt


def _replay_result_key(row):
    """Stable key identifying a sampled row's replay result across runs
    (sample_source::conv_id::turn_id), so results can be resumed/deduped."""
    return "::".join(
        str(row[column])
        for column in ["sample_source", "conv_id", "turn_id"]
    )


def replayer(
    model,
    with_history=False,
    save_every=5,
    output_file=None,
    history_depth=4,
    samples_per_source=1,
    random_seed=RANDOM_SEED,
    replay_provider=None,
    developer_prompt=None,
):
    """Sample rows via filter_df_for_history() and replay each one's most
    recent user prompt against `model`, under every tool_choice mode
    _tool_choices_for_provider returns for its provider (typically just
    "auto"). Results are saved incrementally to `output_file` (every
    `save_every` rows) and can be resumed: rows/tool_choices already present
    in an existing output_file are not re-requested. Returns the full
    {result_key: result} dict.

    `model="invivo"` replays each row under whatever model the platform
    itself used for that turn (from the extracted data) instead of a fixed
    model name.
    """
    resolved_replay_provider = (
        _normalize_provider(replay_provider)
        if replay_provider
        else _infer_provider_from_model(model)
    )
    resolved_developer_prompt = _resolve_developer_prompt(developer_prompt)

    df = filter_df_for_history(
        history_depth,
        samples_per_source=samples_per_source,
        random_seed=random_seed,
    )
    df = df.reset_index(drop=True).copy()
    print("sample counts:", df["sample_source"].value_counts().to_dict())
    print("skip replay sample indices:", sorted(SKIPPED_REPLAY_SAMPLE_INDICES))

    model_results = _load_replay_results(output_file)

    planned_rows = []
    for sample_idx, row in df.iterrows():
        prompt, user_prompt = _build_prompt(
            row["user_msg_history"],
            row["assistant_msg_history"],
            with_history=with_history,
            history_depth=history_depth,
        )
        result_key = _replay_result_key(row)
        duplicate_idx = 2
        while result_key in model_results and model_results[result_key].get("result_key") != result_key:
            result_key = f"{_replay_result_key(row)}::{duplicate_idx}"
            duplicate_idx += 1

        invivo_response = (
            _clean_messages(row["assistant_msg_history"])[-1]
            if _clean_messages(row["assistant_msg_history"])
            else ""
        )
        model_column = "openai_models" if "openai_models" in row.index else "models"
        row_model = _most_frequent_model(row[model_column])
        row_model = model_replacements.get(row_model, row_model)
        row_model = row_model or DEFAULT_MODEL_BY_PROVIDER["openai"]

        base_result = {
            "sample_idx": int(sample_idx),
            "result_key": result_key,
            "user_prompt": user_prompt,
            "prompt": prompt,
            "developer_prompt": developer_prompt,
            "sample_source": row["sample_source"],
            "conv_id": row["conv_id"],
            "turn_id": row["turn_id"],
            "invivo_model": row_model,
            "replay_model": model,
            "replay_provider": resolved_replay_provider,
            "invivo_response": invivo_response,
        }
        existing_result = model_results.get(result_key, {})
        if not isinstance(existing_result, dict):
            existing_result = {}
        existing_result.update(
            {
                key: value
                for key, value in base_result.items()
                if key not in existing_result
            }
        )
        model_results[result_key] = existing_result
        planned_rows.append((sample_idx, row, prompt, result_key, row_model))

    if output_file:
        _save_replay_results(model_results, output_file)

    for idx, (sample_idx, row, prompt, result_key, row_model) in enumerate(
        tqdm(planned_rows, total=len(planned_rows)), start=1
    ):
        if int(sample_idx) in SKIPPED_REPLAY_SAMPLE_INDICES:
            model_results[result_key]["skipped_replay"] = True
            model_results[result_key]["skip_reason"] = (
                f"sample_idx {int(sample_idx)} is in SKIPPED_REPLAY_SAMPLE_INDICES"
            )
            if output_file and save_every and idx % save_every == 0:
                _save_replay_results(model_results, output_file)
            continue

        replay_model = row_model if model == "invivo" else model

        for tool_choice in _tool_choices_for_provider(resolved_replay_provider):
            if tool_choice in model_results[result_key]:
                continue
            # print(tool_choice)
            try:
                payload = _create_response_payload(
                    replay_model,
                    prompt,
                    tool_choice,
                    replay_provider=resolved_replay_provider,
                    developer_prompt=resolved_developer_prompt,
                )
            except Exception as exc:
                payload = {
                    "output_text": "",
                    "response": {},
                    "error": str(exc),
                }
            model_results[result_key][tool_choice] = payload

        if output_file and save_every and idx % save_every == 0:
            _save_replay_results(model_results, output_file)

    if output_file:
        _save_replay_results(model_results, output_file)
    return model_results


if __name__ == "__main__":
    models = [
        # "gpt-5-mini-2025-08-07",
        # "o4-mini-2025-04-16",
        # "gpt-4.1-mini-2025-04-14",
        "gpt-5.3-chat-latest",
        # "grok-4.3",
        # "claude-sonnet-4-6",
        # "deepseek-v4-flash",
    ]
    dev_prompts = [
        "o4-mini",
        "gpt-4.1-mini",
        # "gpt-5.3-chat-latest"
    ]
    for model in models:
        for dp in dev_prompts:
            if model.startswith(dp):
                continue
            print(model, dp)
            output_file = _replay_output_file(model, dp)
            model_results = replayer(
                model,
                with_history=False,
                history_depth=0,
                save_every=5,
                output_file=output_file,
                samples_per_source=500,
                developer_prompt=f"{dp}.md"
            )
            print(len(model_results))
