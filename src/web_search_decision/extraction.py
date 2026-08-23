"""Parses raw ChatGPT, Claude, Grok, and DeepSeek export files into a
per-turn summary dataframe -- the single, unified extraction entry point
for all four platforms (this used to be split into a ChatGPT-only
chatgpt_extraction.py plus this file for the other three; they've been
merged here since the parsing logic is >90% shared and having two loaders
for "the same" pipeline step was a maintenance trap).

Reads data/<platform>/user_<i>/conversations.json (or a flat
data/<platform>/conversations.json for a single-user export; see README.md's
"Exporting Your Own Data") and, for every turn, extracts which tools were
called, thinking-mode flags, message history, timing, and a topic label.

Run directly as:
    python -m src.web_search_decision.extraction --platform claude
(--platform one of: chatgpt, claude, grok, deepseek.)

Output: outputs/<platform>/metadata/data_summary.* + web_data_summary.*,
same layout for all four platforms.

Typical output, run against the paper's own (ERB-restricted, unshared)
donated dataset:
    Claude:   102 users, 9,267 conversations, 64,354 turns, 11,466 with tool calls
    Grok:     100 users, 9,005 conversations, 53,844 turns,  3,291 with tool calls
    DeepSeek: 101 users, 9,262 conversations, 36,020 turns,  1,730 with tool calls
"""

import argparse
import json
import logging
import sys

sys.setrecursionlimit(5000)
from pathlib import Path

import pandas as pd
from langdetect import detect
from tqdm import tqdm

import src.utils.other_platforms_parsing_utils as du
from src.utils.common_io import DATA_BASE_PATH
from src.utils.topic_classifier import classify_topic

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default="data")
    parser.add_argument(
        "--platform",
        type=str,
        required=True,
        choices=["chatgpt", "claude", "grok", "deepseek"],
    )
    parser.add_argument("--output_path", type=Path, default="outputs")
    return parser.parse_args()


# Module-level default so importers (e.g., web_tool_invocation.py) can use
# load_*_from_file without first running main(). main() overrides this.
OUTPUT_PATH = Path("./outputs")


# ---------- Column definitions ----------

# "openai_models" holds the per-message model identifiers regardless of
# platform (kept under this name -- not renamed to something neutral like
# "models" -- because every downstream analysis module already reads
# df["openai_models"] unconditionally; this is the one column name none of
# them may diverge on).
COLUMNS = [
    "user_id",
    "conv_id",
    "turn_id",
    "conv_starter",
    "topic",
    "language",
    "tools",
    "interactions",
    "reasoning",
    "thinking",
    "thoughts",
    "openai_models",
    "user_msg_history",
    "assistant_msg_history",
    "turn_msgs",
    "time",
]


# ---------- Shared helpers ----------

def _detect_language(history):
    """Best-effort language code for a list of message strings, "" on failure."""
    try:
        return detect("\n".join(str(x) for x in history))
    except Exception:
        return ""


def _resolve_topic(topic_lookup, conv_id, user_msg_history):
    """Look up `conv_id`'s annotated topic; if none is available (the common
    case outside the paper's own dataset), classify it from the
    conversation's opening message instead of defaulting to "Other"."""
    topic = topic_lookup.get(conv_id)
    if topic:
        return topic
    first_user_message = user_msg_history[0] if user_msg_history else ""
    return classify_topic(first_user_message)


def _iter_platform_files(data_base_path, conversations_key=None):
    """Yield (user_idx, conversations_list) for a platform export directory.

    Supports two layouts:
    - per-user subdirectories: ``<data_base_path>/user_*/conversations.json``
    - a single flat export: ``<data_base_path>/conversations.json``

    ``conversations_key``, if given, is the key holding the conversation list
    inside the loaded JSON (used by exports like Grok's, whose top-level
    export file is a dict with a "conversations" key rather than a bare list).
    """
    base = Path(data_base_path)

    def _load(file_path):
        with open(file_path) as f:
            data = json.load(f)
        if conversations_key is not None and isinstance(data, dict):
            return data.get(conversations_key) or []
        return data

    user_dirs = sorted(
        base.glob("user_*/conversations.json"),
        key=lambda p: p.parent.name,
    )
    if user_dirs:
        for i, file_path in enumerate(tqdm(user_dirs)):
            yield i, _load(file_path)
        return

    flat_file = base / "conversations.json"
    if flat_file.exists():
        yield 0, _load(flat_file)
        return


# ---------- Web-call detection ----------

GROK_WEB_TOOLS = {
    "WebSearch",
}
DEEPSEEK_WEB_TOOLS = {
    "SEARCH",
}


# ChatGPT's `tools` column holds whatever load_chatgpt_data() pushed into
# main_tool_calls for a turn -- either a legacy recipient name (older/
# plugin-era exports) or the "web_search" marker load_chatgpt_data() injects
# for current exports (see its flush_turn()). Legacy recipient names known
# to mean "the browsing tool ran": exact matches plus a substring check,
# since the exact identifier drifted across ChatGPT's tool-naming eras
# (plugin-era "browser", later "web"/"web.run", ...) and we don't have
# enough historical exports on hand to enumerate every variant precisely.
_CHATGPT_LEGACY_WEB_RECIPIENTS = {"browser", "web", "web.run"}
_CHATGPT_MODERN_WEB_MARKER = "web_search"


def _chatgpt_has_web_call(tools):
    """Two-step web-search detection for one ChatGPT turn's `tools` list,
    since the export wire format for signaling it changed over time:

    step 1 (legacy/plugin-era exports) -- the turn contains a message
    explicitly routed to a known browsing-tool recipient name;
    step 2 (current/2025+ exports) -- no such message exists; instead the
    turn contains the "web_search" marker load_chatgpt_data() injects from
    metadata.search_result_groups / a thoughts block's tool_icons.

    Old exports are only ever expected to satisfy step 1, current exports
    only step 2 -- but both are checked unconditionally, in case a single
    account's export spans both eras.
    """
    for tool in tools or []:
        tool_lower = str(tool).lower()
        if tool_lower == _CHATGPT_MODERN_WEB_MARKER:
            return True  # step 2: current export format
        if tool_lower in _CHATGPT_LEGACY_WEB_RECIPIENTS or "web" in tool_lower:
            return True  # step 1: legacy/plugin-era export format
    return False


def web_call_mask(df, platform):
    """Boolean Series flagging turns that invoked a Web-search tool.

    Each platform exposes tool usage differently (ChatGPT: see
    _chatgpt_has_web_call's two-step check on the `tools` column; Claude:
    tool names, any containing "web"; Grok/DeepSeek: fixed tool-name sets),
    so this centralizes the per-platform heuristic previously duplicated
    inline in main()'s web_data_summary split.
    """
    if platform == "chatgpt":
        return df["tools"].apply(_chatgpt_has_web_call)
    elif platform == "claude":
        return df["tools"].apply(
            lambda ts: any("web" in str(t).lower() for t in (ts or []))
        )
    elif platform == "grok":
        return df["tools"].apply(
            lambda ts: any(t in GROK_WEB_TOOLS for t in (ts or []))
        )
    elif platform == "deepseek":
        return df["tools"].apply(
            lambda ts: any(t in DEEPSEEK_WEB_TOOLS for t in (ts or []))
        )
    raise ValueError(
        f"Unknown platform: {platform!r}. "
        "Use 'chatgpt', 'claude', 'grok', or 'deepseek'."
    )


# ---------- Dispatch wrapper ----------

def load_whole_data(platform):
    """Extract every user's conversations.json for `platform` into a list of
    per-turn rows (see COLUMNS). Dispatches to the platform-specific loader."""
    if platform == "chatgpt":
        return load_chatgpt_data()
    elif platform == "claude":
        return load_claude_data()
    elif platform == "grok":
        return load_grok_data()
    elif platform == "deepseek":
        return load_deepseek_data()
    raise ValueError(
        f"Unknown platform: {platform!r}. "
        "Use 'chatgpt', 'claude', 'grok', or 'deepseek'."
    )


# ---------- ChatGPT ----------

def _iter_chatgpt_files():
    """Yield (user_idx, conversations_list) for ChatGPT export files."""
    yield from _iter_platform_files(DATA_BASE_PATH)


def load_chatgpt_data():
    """Extract every user's ChatGPT conversations.json into per-turn rows."""
    all_data = []
    num_users = 0
    num_conversations = 0
    num_turns = 0
    num_msgs = 0
    num_tool_usage = 0
    tools = []
    topic_lookup = du.load_topics("chatgpt")

    for i, data in _iter_chatgpt_files():
        num_users += 1
        for conv in data:
            num_conversations += 1
            mapping = du.sort_conversation(conv["mapping"], "chatgpt")

            user_msg_history = []
            assistant_msg_history = []
            turn_msgs = []
            conv_starter = 1

            def flush_turn():
                """Emit one row per turn."""
                nonlocal num_turns, num_msgs, num_tool_usage, conv_starter
                if not turn_msgs:
                    return

                num_turns += 1
                main_tool_calls = []
                reasoning_path = []
                thinking_path = []
                models = []
                interactions = []
                thoughts = ""
                turn_has_modern_web_search = False  # dedup guard, see below

                for turn_msg in turn_msgs:
                    author = turn_msg.get("author", {})
                    role_ = author.get("name") or author.get("role", "")

                    # Missing "recipient" means the message went to the user
                    # (ChatGPT only sets it explicitly when routing to a tool).
                    recipient = turn_msg.get("recipient", "all")
                    ts = turn_msg.get("create_time")
                    metadata_ = turn_msg.get("metadata", {}) or {}
                    models.append(metadata_.get("model_slug"))
                    reasoning_path.append(metadata_.get("reasoning_status"))

                    thinking_type = turn_msg.get("content", {}).get("content_type")
                    thinking_path.append(thinking_type)
                    if thinking_type == "thoughts":
                        for tt in turn_msg["content"].get("thoughts", []):
                            thoughts += tt.get("content", "") + "\n"

                    interactions.append(f"{role_}:{recipient}")
                    if ts and role_ == "assistant" and recipient != "all":
                        # Older/plugin-era export format: the assistant
                        # explicitly routes the message to a named tool
                        # (recipient="browser", "web", "dalle.text2im", ...).
                        main_tool_calls.append(recipient)
                        num_tool_usage += 1
                    elif (
                        ts
                        and role_ == "assistant"
                        and not turn_has_modern_web_search
                        and (
                            metadata_.get("search_result_groups")
                            or (thinking_type == "thoughts" and metadata_.get("tool_icons"))
                        )
                    ):
                        # Current (2025+) export format has no separate
                        # tool-routed message -- there are two signals a web
                        # search happened instead, checked in order:
                        # (a) the assistant's answer message metadata carries
                        #     actual retrieved results (search_result_groups) --
                        #     same signal source_selection.py/
                        #     response_generation.py/query_reformulations.py
                        #     already read to extract retrieved sources;
                        # (b) a "thoughts" block with non-empty tool_icons
                        #     (site favicons) -- fires even when (a) is
                        #     absent, e.g. a single-page fetch that never
                        #     populates search_result_groups.
                        # `turn_has_modern_web_search` stops a turn whose
                        # thoughts *and* answer message both carry the
                        # signal from being counted as two tool calls.
                        main_tool_calls.append("web_search")
                        num_tool_usage += 1
                        interactions.append(f"{role_}:web_search")
                        turn_has_modern_web_search = True

                reasoning = any(reasoning_path)
                thinking = "thoughts" in thinking_path
                time_ = du.normalize_timestamp(turn_msgs[-1].get("create_time"), "chatgpt")
                language = _detect_language(user_msg_history)
                topic = _resolve_topic(topic_lookup, conv["id"], user_msg_history)

                all_data.append([
                    i,
                    conv["id"],
                    turn_msgs[0]["id"],
                    conv_starter,
                    topic,
                    language,
                    main_tool_calls,
                    interactions,
                    int(reasoning),
                    int(thinking),
                    thoughts,
                    models,
                    user_msg_history.copy(),
                    assistant_msg_history.copy(),
                    json.dumps(turn_msgs),
                    time_,
                ])
                tools.extend(main_tool_calls)
                conv_starter = 0
                num_msgs += len(turn_msgs)

            for msg_info in mapping:
                msg = msg_info["raw"].get("message")
                if not msg:
                    continue
                role = msg.get("author", {}).get("role", "")
                if role == "system":
                    continue

                # New user message ends the previous turn.
                if role == "user" and turn_msgs:
                    flush_turn()
                    turn_msgs = []

                turn_msgs.append(msg)
                if role == "user":
                    parts = msg.get("content", {}).get("parts", [])
                    user_msg_history += [str(x) for x in parts]
                if role == "assistant" and msg.get("recipient", "all") == "all":
                    parts = msg.get("content", {}).get("parts", [])
                    assistant_msg_history += [str(x) for x in parts]

            # Flush trailing turn
            flush_turn()

    return all_data, num_users, num_conversations, num_turns, num_msgs, num_tool_usage


# ---------- Claude ----------

def _iter_claude_files():
    """Yield (user_idx, conversations_list) for Claude export files."""
    yield from _iter_platform_files(DATA_BASE_PATH)


def load_claude_data():
    """Extract every user's Claude conversations.json into per-turn rows."""
    all_data = []
    num_users = 0
    num_conversations = 0
    num_turns = 0
    num_msgs = 0
    num_tool_usage = 0
    tools = []
    topic_lookup = du.load_topics("claude")

    for i, data in _iter_claude_files():
        num_users += 1
        for conv in data:
            num_conversations += 1
            sorted_msgs = du.sort_conversation(conv["chat_messages"], "claude")

            user_msg_history = []
            assistant_msg_history = []
            turn_msgs = []
            conv_starter = 1

            def flush_turn():
                """Emit one row per turn."""
                nonlocal num_turns, num_msgs, num_tool_usage, conv_starter
                if not turn_msgs:
                    return

                num_turns += 1
                tool_calls = []
                interactions = []
                thinking_path = []
                thoughts = ""

                for m in turn_msgs:
                    sender = m.get("sender")
                    for b in m.get("content") or []:
                        btype = b.get("type")
                        thinking_path.append(btype)
                        if btype == "tool_use":
                            tool_calls.append(b.get("name"))
                            num_tool_usage += 1
                        elif btype == "thinking":
                            thoughts += b.get("thinking", "") + "\n"
                    interactions.append(sender)

                thinking = "thinking" in thinking_path
                time_ = du.normalize_timestamp(turn_msgs[-1].get("created_at"), "claude")
                language = _detect_language(user_msg_history)
                topic = _resolve_topic(topic_lookup, conv["uuid"], user_msg_history)

                all_data.append([
                    i,
                    conv["uuid"],
                    turn_msgs[0]["uuid"],
                    conv_starter,
                    topic,
                    language,
                    tool_calls,
                    interactions,
                    0,  # reasoning not explicitly labeled in Claude data
                    int(thinking),
                    thoughts,
                    [],  # no model info in Claude data
                    user_msg_history.copy(),
                    assistant_msg_history.copy(),
                    json.dumps(turn_msgs, default=str),
                    time_,
                ])
                tools.extend(tool_calls)
                conv_starter = 0
                num_msgs += len(turn_msgs)

            for msg_info in sorted_msgs:
                msg = msg_info["raw"]
                sender = msg.get("sender")

                # New human message ends the previous turn
                if sender == "human" and turn_msgs:
                    flush_turn()
                    turn_msgs = []

                turn_msgs.append(msg)
                blocks = msg.get("content") or []
                text_parts = [
                    b.get("text", "") for b in blocks
                    if b.get("type") == "text" and b.get("text")
                ]
                if not text_parts and msg.get("text"):
                    text_parts = [msg["text"]]

                if sender == "human":
                    user_msg_history += text_parts
                elif sender == "assistant":
                    assistant_msg_history += text_parts

            # Flush trailing turn
            flush_turn()

    return all_data, num_users, num_conversations, num_turns, num_msgs, num_tool_usage


# ---------- Grok ----------

def _iter_grok_files():
    """Yield (user_idx, sessions_list) for Grok export files.

    Grok's export file is a dict with a top-level "conversations" key
    (alongside "projects", "tasks", "media_posts"), not a bare list.
    """
    yield from _iter_platform_files(DATA_BASE_PATH, conversations_key="conversations")


def load_grok_data():
    """Extract every user's Grok conversations.json into per-turn rows."""
    all_data = []
    num_users = 0
    num_conversations = 0
    num_turns = 0
    num_msgs = 0
    num_tool_usage = 0
    tools = []
    topic_lookup = du.load_topics("grok")

    for i, sessions in _iter_grok_files():
        num_users += 1
        for session in sessions:
            num_conversations += 1
            conv_meta = session.get("conversation") or {}
            sorted_msgs = du.sort_conversation(session, "grok")

            user_msg_history = []
            assistant_msg_history = []
            turn_msgs = []
            conv_starter = 1

            def flush_turn():
                """Emit one row per turn."""
                nonlocal num_turns, num_msgs, num_tool_usage, conv_starter
                if not turn_msgs:
                    return

                num_turns += 1
                tool_calls = []
                interactions = []
                models = []
                thoughts = ""
                seen_cards = set()  # dedup across steps within the turn

                for m in turn_msgs:
                    sender = m.get("sender")
                    interactions.append(sender)
                    model = m.get("model")
                    if model:
                        models.append(model)

                    for step in m.get("steps") or []:
                        # Collect tagged_text.summary as the reasoning trace
                        # (Grok exports per-step summary text; the closest analog
                        # of "thoughts" we have for Grok).
                        summary_text = (
                            (step.get("tagged_text") or {}).get("summary") or ""
                        )
                        if summary_text:
                            thoughts += summary_text + "\n"

                        for card in step.get("tool_usage_cards") or []:
                            card_id = card.get("tool_usage_card_id")
                            if card_id is not None:
                                if card_id in seen_cards:
                                    continue
                                seen_cards.add(card_id)
                            tool_obj = card.get("tool") or {}
                            for tool_name in tool_obj.keys():
                                tool_calls.append(tool_name)
                                num_tool_usage += 1

                thinking = 1 if thoughts else 0

                time_ = du.normalize_timestamp(turn_msgs[-1].get("create_time"), "grok")
                language = _detect_language(user_msg_history)
                topic = _resolve_topic(topic_lookup, conv_meta.get("id"), user_msg_history)

                all_data.append([
                    i,
                    conv_meta.get("id"),
                    turn_msgs[0].get("_id"),
                    conv_starter,
                    topic,
                    language,
                    tool_calls,
                    interactions,
                    0,  # reasoning not separately labeled in Grok export
                    thinking,
                    thoughts,
                    models,
                    user_msg_history.copy(),
                    assistant_msg_history.copy(),
                    json.dumps(turn_msgs, default=str),
                    time_,
                ])
                tools.extend(tool_calls)
                conv_starter = 0
                num_msgs += len(turn_msgs)

            for msg_info in sorted_msgs:
                msg = msg_info["raw"]
                sender = msg.get("sender")

                # New human message ends the previous turn
                if sender == "human" and turn_msgs:
                    flush_turn()
                    turn_msgs = []

                turn_msgs.append(msg)
                if sender == "human":
                    text = msg.get("message", "") or ""
                    if text:
                        user_msg_history.append(text)
                # ASSISTANT text (steps/tagged_text) intentionally skipped —
                # not needed for web-call analysis.

            # Flush trailing turn
            flush_turn()

    return all_data, num_users, num_conversations, num_turns, num_msgs, num_tool_usage


# ---------- DeepSeek ----------

def _iter_deepseek_files():
    """Yield (user_idx, conversations_list) for DeepSeek export files."""
    yield from _iter_platform_files(DATA_BASE_PATH)


def load_deepseek_data():
    """Extract every user's DeepSeek conversations.json into per-turn rows."""
    all_data = []
    num_users = 0
    num_conversations = 0
    num_turns = 0
    num_msgs = 0
    num_tool_usage = 0
    tools = []
    topic_lookup = du.load_topics("deepseek")

    TEXT_TYPES = {"REQUEST", "RESPONSE", "THINK"}

    for i, data in _iter_deepseek_files():
        num_users += 1
        for conv in data:
            num_conversations += 1
            sorted_msgs = du.sort_conversation(conv, "deepseek")

            user_msg_history = []
            assistant_msg_history = []
            turn_msgs = []  # list of node dicts
            conv_starter = 1

            def flush_turn():
                """Emit one row per turn."""
                nonlocal num_turns, num_msgs, num_tool_usage, conv_starter
                if not turn_msgs:
                    return

                num_turns += 1
                tool_calls = []
                interactions = []
                models = []
                thoughts = ""
                thinking = 0

                for node in turn_msgs:
                    msg = node.get("message") or {}
                    model = msg.get("model")
                    if model:
                        models.append(model)

                    fragments = msg.get("fragments") or []
                    files = msg.get("files") or []
                    has_request = (
                        any(f.get("type") == "REQUEST" for f in fragments)
                        or (not fragments and bool(files))  # file-only user msg
                    )
                    sender = "user" if has_request else "assistant"
                    interactions.append(sender)

                    for f in fragments:
                        ftype = f.get("type")
                        if ftype == "THINK":
                            thoughts += (f.get("content") or "") + "\n"
                            thinking = 1
                        elif ftype not in TEXT_TYPES:
                            tool_calls.append(ftype)
                            num_tool_usage += 1

                last_msg = (turn_msgs[-1].get("message") or {})
                time_ = du.normalize_timestamp(
                    last_msg.get("inserted_at"), "deepseek"
                )
                language = _detect_language(user_msg_history)
                topic = _resolve_topic(topic_lookup, conv.get("id"), user_msg_history)

                all_data.append([
                    i,
                    conv.get("id"),
                    turn_msgs[0].get("id"),
                    conv_starter,
                    topic,
                    language,
                    tool_calls,
                    interactions,
                    0,  # reasoning not explicitly flagged
                    int(thinking),
                    thoughts,
                    models,
                    user_msg_history.copy(),
                    assistant_msg_history.copy(),
                    json.dumps(turn_msgs, default=str),
                    time_,
                ])
                tools.extend(tool_calls)
                conv_starter = 0
                num_msgs += len(turn_msgs)

            for msg_info in sorted_msgs:
                node = msg_info["raw"]
                msg = node.get("message") or {}
                fragments = msg.get("fragments") or []
                files = msg.get("files") or []
                # A user "REQUEST" is either an explicit REQUEST fragment OR a
                # file-only user msg (empty fragments but with attached files).
                # no_files empty nodes (USER<->USER stubs) are NOT turn boundaries.
                has_request = (
                    any(f.get("type") == "REQUEST" for f in fragments)
                    or (not fragments and bool(files))
                )

                # New user message ends previous turn
                if has_request and turn_msgs:
                    flush_turn()
                    turn_msgs = []

                turn_msgs.append(node)

                if has_request:
                    for f in fragments:
                        if f.get("type") == "REQUEST":
                            content = f.get("content") or ""
                            if content:
                                user_msg_history.append(content)
                else:
                    for f in fragments:
                        if f.get("type") == "RESPONSE":
                            content = f.get("content") or ""
                            if content:
                                assistant_msg_history.append(content)

            # Flush trailing turn
            flush_turn()

    return all_data, num_users, num_conversations, num_turns, num_msgs, num_tool_usage


def _metadata_dir(platform):
    """outputs/<platform>/metadata for every platform, ChatGPT included.

    load_whole_data_from_file()/load_web_data_from_file() below are the
    single choke point every downstream module (source_selection.py,
    response_generation.py, query_reformulations.py, chat_replayer.py,
    extract_replay_artifacts.py, web_tool_invocation.py) uses to read
    extracted data, so keeping this one function symmetric is enough to
    make the whole raw-extraction layer consistent across platforms --
    nothing downstream needs its own per-platform path logic for this
    step. (A handful of *other*, unrelated artifacts further down the
    pipeline -- response_and_sources.pkl and beyond -- have their own,
    separately-tracked platform handling; see README's "Pipeline Order &
    Known Gaps".)
    """
    return f"{OUTPUT_PATH}/{platform}/metadata"


def load_whole_data_from_file(fmt, platform="chatgpt"):
    """Load a previously extracted data_summary.<fmt> back into a DataFrame."""
    base_dir = _metadata_dir(platform)
    if fmt == "csv":
        df = pd.read_csv(f"{base_dir}/data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(f"{base_dir}/data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(f"{base_dir}/data_summary.parquet")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return df


def load_web_data_from_file(fmt, platform="chatgpt"):
    """Load a previously extracted web_data_summary.<fmt> back into a DataFrame
    (data_summary.* filtered down to turns that invoked a Web-search tool)."""
    base_dir = _metadata_dir(platform)
    if fmt == "csv":
        df = pd.read_csv(f"{base_dir}/web_data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(f"{base_dir}/web_data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(f"{base_dir}/web_data_summary.parquet")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return df


def main():
    """CLI entry point: extract one platform's conversations and write
    data_summary.* / web_data_summary.* under its metadata dir (see
    _metadata_dir)."""
    global DATA_BASE_PATH, OUTPUT_PATH
    args = parse_args()
    DATA_BASE_PATH = Path(args.base_dir) / args.platform
    OUTPUT_PATH = Path(args.output_path)
    base_dir = _metadata_dir(args.platform)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    all_data, num_users, num_conversations, num_turns, num_msgs, num_tool_usage = load_whole_data(
        args.platform
    )

    # Convert to DataFrame
    df = pd.DataFrame(
        all_data,
        columns=COLUMNS,
    )
    df = df.sort_values("time")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

    print(f"Number of Users: {num_users}")
    print(f"Number of Conversation: {num_conversations}")
    print(f"Number of Turns: {num_turns}")
    print(f"Number of Turns with Tool calls: {len(df[df['tools'].apply(len) > 0])}")
    print(f"Number of Messages: {num_msgs}")
    print(f"Number of Messages with Tool calls: {num_tool_usage}")

    df = df.reset_index(drop=True)
    df.to_parquet(f"{base_dir}/data_summary.parquet")
    df.to_pickle(f"{base_dir}/data_summary.pkl")
    df.to_csv(f"{base_dir}/data_summary.csv", index=False)
    print("All Data Saved Successfully!")

    web_df = df[web_call_mask(df, args.platform)]

    web_df = web_df.reset_index(drop=True)
    web_df.to_parquet(f"{base_dir}/web_data_summary.parquet")
    web_df.to_pickle(f"{base_dir}/web_data_summary.pkl")
    web_df.to_csv(f"{base_dir}/web_data_summary.csv", index=False)
    print("Web Data Saved Successfully!")


if __name__ == "__main__":
    main()
