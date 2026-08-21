import json
import ast
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
from src.utils import *
import pandas as pd
from src.data_extraction import load_whole_data_from_file, load_web_data_from_file
from tqdm import tqdm
import numpy as np

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
    from openai import OpenAI

    return OpenAI


def _import_anthropic():
    from anthropic import Anthropic

    return Anthropic


def _normalize_provider(provider):
    key = (provider or "openai").strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def _resolve_developer_prompt(developer_prompt):
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
    model_key = (model or "").strip().lower()
    if model_key.startswith("claude"):
        return "claude"
    if model_key.startswith("grok"):
        return "grok"
    if model_key.startswith("deepseek"):
        return "deepseek"
    return "openai"


def _openai_compatible_base_url(provider):
    env_base_urls = {
        "openai": os.getenv("OPENAI_BASE_URL", "").strip(),
        "grok": os.getenv("XAI_BASE_URL", "").strip() or "https://api.x.ai/v1",
    }
    return env_base_urls.get(provider) or None


def _client_for_provider(provider):
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
        if provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com/anthropic"
        if base_url:
            kwargs["base_url"] = base_url
        return Anthropic(**kwargs)

    raise ValueError(f"unsupported replay provider: {provider}")


def _as_list(value):
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
    if history_depth is None:
        return None
    return max(history_depth // 2, 0)

def _clean_messages(messages):
    return [str(msg).strip() for msg in _as_list(messages) if str(msg).strip()]


def _has_exact_history_depth(row, prior_turns):
    user_prompts = _clean_messages(row["user_msg_history"])
    assistant_prompts = _clean_messages(row["assistant_msg_history"])
    return (
        len(user_prompts) == prior_turns + 1
        and len(assistant_prompts) >= prior_turns
    )


def _safe_annotation_conv_ids(annotation_path=ANNOTATIONS_TURNS_PATH):
    annotations = pd.read_csv(
        annotation_path,
        usecols=lambda column: column in ANNOTATION_REQUIRED_COLUMNS,
        dtype=str,
    )
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
    return df[df["conv_id"].astype(str).isin(safe_conv_ids)].copy()


def filter_df_for_history(
    history_depth=0,
    samples_per_source=1,
    random_seed=RANDOM_SEED,
):
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
    return tool_choices


def _openai_web_kwargs(
    replay_model,
    prompt,
    tool_choice,
    developer_prompt=None,
):
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


def _openai_chat_completion_payload(response):
    message = response.choices[0].message
    content = message.content
    if isinstance(content, list):
        output_text = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    else:
        output_text = (content or "").strip()
    return {
        "output_text": output_text,
        "response": response.model_dump(),
    }


def _anthropic_content_text(content_blocks):
    texts = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            texts.append(getattr(block, "text", ""))
    return "\n".join(text for text in texts if text).strip()


def _anthropic_web_tools(tool_choice):
    if tool_choice == "none":
        return None
    return [
        {
            "type": "web_search_20250305",
            "name": "web_search",
        }
    ]


def _anthropic_tool_choice(tool_choice):
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
    kwargs = {
        "model": replay_model,
        "messages": prompt,
        "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
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
    valid_models = [
        m
        for m in openai_models
        if isinstance(m, str) and m.strip() and m.lower() != "none"
    ]
    if not valid_models:
        return None
    return Counter(valid_models).most_common(1)[0][0]


def _save_replay_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def _load_replay_results(output_file):
    if not output_file:
        return {}
    path = Path(output_file)
    if not path.exists() or not path.is_file():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _replay_output_file(model, dev_prompt=""):
    if dev_prompt:
        return f"{OUTPUT_PATH}/replays/{model}_dev_prompt_{dev_prompt}.json"
    return f"{OUTPUT_PATH}/replays/{model}.json"


def _build_prompt(user_prompts, assistant_prompts, with_history, history_depth):
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
