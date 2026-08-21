import argparse
import ast
import json
import sys

sys.setrecursionlimit(5000)
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import pandas as pd
from urllib.parse import urlparse
from langdetect import detect

import src.data_utils_cai as du
from src.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default="data")
    parser.add_argument("--platform", type=str, default="claude")
    parser.add_argument("--output_path", type=Path, default="outputs")
    return parser.parse_args()


# Module-level default so importers (e.g., web_tool_invocation.py) can use
# load_*_from_file without first running main(). main() overrides this.
OUTPUT_PATH = Path("./outputs")


# ---------- Column definitions ----------


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
    "other_platforms_models",
    "user_msg_history",
    "assistant_msg_history",
    "turn_msgs",
    "time",
]


# ---------- Shared helpers ----------

def _detect_language(history):
    try:
        return detect("\n".join(str(x) for x in history))
    except Exception:
        return ""


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
    "WebSearch", "BrowsePage", "XThreadFetch", "XSearch",
    "XUserSearch", "ViewXVideo", "ImageSearch", "PdfSearch", "PdfBrowse",
}
DEEPSEEK_WEB_TOOLS = {
    "SEARCH", "READ_LINK", "TOOL_SEARCH", "TOOL_OPEN", "TOOL_FIND",
}


def web_call_mask(df, platform):
    """Boolean Series flagging turns that invoked a Web-search tool.

    Each platform exposes tool usage differently (ChatGPT: free-text
    "interactions" trace; Claude: tool names, any containing "web"; Grok/
    DeepSeek: fixed tool-name sets), so this centralizes the per-platform
    heuristic previously duplicated inline in main()'s web_data_summary split.
    """
    if platform == "chatgpt":
        return df["interactions"].apply(lambda x: "web" in str(x))
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

                for turn_msg in turn_msgs:
                    author = turn_msg.get("author", {})
                    role_ = author.get("name") or author.get("role", "")

                    # Missing "recipient" means the message went to the user
                    # (ChatGPT only sets it explicitly when routing to a tool).
                    recipient = turn_msg.get("recipient", "all")
                    ts = turn_msg.get("create_time")
                    models.append(turn_msg.get("metadata", {}).get("model_slug"))
                    reasoning_path.append(
                        turn_msg.get("metadata", {}).get("reasoning_status")
                    )

                    thinking_type = turn_msg.get("content", {}).get("content_type")
                    thinking_path.append(thinking_type)
                    if thinking_type == "thoughts":
                        for tt in turn_msg["content"].get("thoughts", []):
                            thoughts += tt.get("content", "") + "\n"

                    interactions.append(f"{role_}:{recipient}")
                    if ts and role_ == "assistant" and recipient != "all":
                        main_tool_calls.append(recipient)
                        num_tool_usage += 1

                reasoning = any(reasoning_path)
                thinking = "thoughts" in thinking_path
                time_ = du.normalize_timestamp(turn_msgs[-1].get("create_time"), "chatgpt")
                language = _detect_language(user_msg_history)

                all_data.append([
                    i,
                    conv["id"],
                    turn_msgs[0]["id"],
                    conv_starter,
                    topic_lookup.get(conv["id"], "Other"),
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

                all_data.append([
                    i,
                    conv["uuid"],
                    turn_msgs[0]["uuid"],
                    conv_starter,
                    topic_lookup.get(conv["uuid"], "Other"),
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

                all_data.append([
                    i,
                    conv_meta.get("id"),
                    turn_msgs[0].get("_id"),
                    conv_starter,
                    topic_lookup.get(conv_meta.get("id"), "Other"),
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

                all_data.append([
                    i,
                    conv.get("id"),
                    turn_msgs[0].get("id"),
                    conv_starter,
                    topic_lookup.get(conv.get("id"), "Other"),
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


def load_whole_data_from_file(fmt, platform):
    base_dir = f"{OUTPUT_PATH}/{platform}/metadata"
    if fmt == "csv":
        df = pd.read_csv(f"{base_dir}/data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(f"{base_dir}/data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(f"{base_dir}/data_summary.parquet")
    return df


def load_web_data_from_file(fmt, platform):
    base_dir = f"{OUTPUT_PATH}/{platform}/metadata"
    if fmt == "csv":
        df = pd.read_csv(f"{base_dir}/web_data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(f"{base_dir}/web_data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(f"{base_dir}/web_data_summary.parquet")
    return df


def main():
    global DATA_BASE_PATH, OUTPUT_PATH
    args = parse_args()
    DATA_BASE_PATH = Path(args.base_dir) / args.platform
    OUTPUT_PATH = Path(args.output_path)
    base_dir = f"{OUTPUT_PATH}/{args.platform}/metadata"
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


# # CLAUDE OUTPUT:
# Number of Users: 102
# Number of Conversation: 9267
# Number of Turns: 64354
# Number of Turns with Tool calls: 11466
# Number of Messages: 137626
# Number of Messages with Tool calls: 36074

# # Grok OUTPUT:
# Number of Users: 100                                                                                              
# Number of Conversation: 9005                                                                                      
# Number of Turns: 53844                                                                                                                                                                                                              
# Number of Turns with Tool calls: 3291
# Number of Messages: 110670                                                                                        
# Number of Messages with Tool calls: 22762

# # DeepSeek OUTPUT:
# Number of Users: 101
# Number of Conversation: 9262
# Number of Turns: 36020
# Number of Turns with Tool calls: 1730
# Number of Messages: 72024
# Number of Messages with Tool calls: 2231