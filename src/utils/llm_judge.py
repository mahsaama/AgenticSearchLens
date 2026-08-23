"""Shared multi-provider LLM-judge client.

Each platform's own data is judged by that platform's own model -- Claude
Haiku 3.5 for Claude, Grok-3-mini for Grok, DeepSeek-chat for DeepSeek,
gpt-4o-mini for ChatGPT -- rather than a single fixed judge shared across
all four. This is a deliberate choice (made explicitly, not a default):
using each platform's own model as its judge means judged metrics
(specificity, factuality, tool-intent, etc.) are NOT directly comparable
across platforms the way a single fixed judge would make them, but it
avoids picking one platform's model family as the arbiter for everyone
else's data.

Reuses the same provider/API-shape split src/replays/chat_replayer.py's
invitro replay already established: OpenAI/Grok via the OpenAI-compatible
Responses API, Claude/DeepSeek via the Anthropic-compatible Messages API.
Most judge calls are tool-free (a judge should only classify the text
it's given, never browse) -- pass require_web_search=True for the rare
judge that specifically needs live web search to do its job (e.g. a
factuality check), which reuses the exact same per-provider web_search
tool wiring chat_replayer.py's invitro replay already uses.
"""

import ast
import json
import os

from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL_BY_PLATFORM = {
    "chatgpt": "gpt-4o-mini",
    # Claude 3.5 Haiku is no longer served by the Anthropic API (retired);
    # claude-haiku-4-5 is the current Haiku-tier model.
    "claude": "claude-haiku-4-5-20251001",
    "grok": "grok-3-mini",
    "deepseek": "deepseek-chat",
}

_OPENAI_COMPATIBLE_PLATFORMS = {"chatgpt", "grok"}
_ANTHROPIC_COMPATIBLE_PLATFORMS = {"claude", "deepseek"}

_API_KEY_ENV_BY_PLATFORM = {
    "chatgpt": "OPENAI_API_KEY",
    "grok": "XAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
# Optional base-URL overrides -- empty/unset means "use the SDK's own
# default endpoint" (see chat_replayer._openai_compatible_base_url and the
# .env comment on OPENAI_BASE_URL/ANTHROPIC_BASE_URL for why these must be
# genuinely unset, not `KEY=`, to actually fall through to that default).
_BASE_URL_ENV_BY_PLATFORM = {
    "chatgpt": "OPENAI_BASE_URL",
    "grok": "XAI_BASE_URL",
    "claude": "ANTHROPIC_BASE_URL",
    "deepseek": "DEEPSEEK_ANTHROPIC_BASE_URL",
}

_clients = {}  # one cached, authenticated client instance per platform


def judge_model_for_platform(platform):
    """The judge model this platform's own data should be scored by."""
    if platform not in JUDGE_MODEL_BY_PLATFORM:
        raise ValueError(
            f"No judge model configured for platform {platform!r}; "
            f"expected one of {sorted(JUDGE_MODEL_BY_PLATFORM)}."
        )
    return JUDGE_MODEL_BY_PLATFORM[platform]


def _client_for_platform(platform):
    if platform in _clients:
        return _clients[platform]

    api_key = os.getenv(_API_KEY_ENV_BY_PLATFORM[platform])
    base_url = (
        os.getenv(_BASE_URL_ENV_BY_PLATFORM.get(platform, ""), "").strip() or None
    )

    if platform in _OPENAI_COMPATIBLE_PLATFORMS:
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
    elif platform in _ANTHROPIC_COMPATIBLE_PLATFORMS:
        from anthropic import Anthropic

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = Anthropic(**kwargs)
    else:
        raise ValueError(f"Unsupported judge platform: {platform!r}")

    _clients[platform] = client
    return client


def _parse_judge_json(text):
    """Best-effort JSON extraction from a judge's raw text response (which
    may be bare JSON, JSON wrapped in prose, or markdown-fenced).

    Finds the first balanced {...} object via brace-depth counting, not a
    naive first-`{`-to-last-`}` slice: that breaks when a provider's
    response text concatenates more than one JSON blob together -- seen in
    practice from xAI's Responses API under forced multi-round tool use,
    where output_text can hold several {...} fragments plus stray
    continuation tokens back to back.
    """
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return ast.literal_eval(candidate)
                        except (ValueError, SyntaxError):
                            pass
                    break
        start = text.find("{", start + 1)
    return {}


def _anthropic_content_text(content_blocks):
    texts = [
        getattr(block, "text", "")
        for block in content_blocks
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(text for text in texts if text).strip()


def run_judge(
    platform,
    system_prompt,
    user_prompt,
    temperature=0,
    max_tokens=1024,
    require_web_search=False,
):
    """Run one judge call using `platform`'s own model (see
    JUDGE_MODEL_BY_PLATFORM).

    Tool-free by default (`require_web_search=False`): the judge only
    classifies the given text. Set `require_web_search=True` for a judge
    that needs live web search to do its job (e.g. checking a claim
    against current web content) -- each provider's own web_search tool is
    forced on, and `temperature` is omitted (matching how
    entailment_analysis.py's factuality judge already called the OpenAI
    Responses API: providers don't consistently accept a temperature
    override alongside a forced tool call).

    Returns {"raw_judgment": str, "parsed_judgment": dict, "model": str}
    -- the first two keys match the shape the single-fixed-judge helpers
    this replaces returned (e.g. query_reformulations._run_judge,
    entailment_analysis.evaluate_claim_factuality), so callers only need
    to change how they invoke the judge (pass `platform` instead of a
    pre-built client + model_name), not how they consume the result.
    """
    model_name = judge_model_for_platform(platform)
    client = _client_for_platform(platform)

    if platform in _OPENAI_COMPATIBLE_PLATFORMS:
        kwargs = {
            "model": model_name,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if require_web_search:
            # Same web_search tool shape chat_replayer.py's invitro replay
            # uses for OpenAI/Grok (_openai_web_kwargs).
            kwargs["tools"] = [{"type": "web_search"}]
            kwargs["tool_choice"] = "required"
            kwargs["max_output_tokens"] = max_tokens
        else:
            # Tool-free by omission (not `tools=[], tool_choice="none"`):
            # xAI's OpenAI-compatible endpoint rejects tool_choice when no
            # tools are given ("tool_choice was set but no tools were
            # specified"), while omitting both is tool-free on every
            # OpenAI-compatible provider.
            kwargs["temperature"] = temperature
        response = client.responses.create(**kwargs)
        raw_text = response.output_text
    else:
        kwargs = {
            "model": model_name,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
        }
        if require_web_search:
            # Same web_search tool shape chat_replayer.py's invitro replay
            # uses for Claude/DeepSeek. tool_choice is "auto", not a forced
            # {"type": "tool", "name": "web_search"}: forcing it holds for
            # every turn of a multi-round server-side tool-use exchange, not
            # just the first, so a forced choice can never let the model
            # stop searching and answer -- confirmed via DeepSeek looping
            # through 7 rounds of web_search and never producing a final
            # answer under a forced choice, vs. converging in one search
            # under "auto". Since web_search is the only tool offered and
            # the system prompt already instructs using it, "auto" still
            # searches in practice while allowing the model to conclude.
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
            kwargs["tool_choice"] = {"type": "auto"}
        else:
            kwargs["temperature"] = temperature
        response = client.messages.create(**kwargs)
        raw_text = _anthropic_content_text(response.content)

    return {
        "raw_judgment": raw_text,
        "parsed_judgment": _parse_judge_json(raw_text),
        "model": model_name,
    }
