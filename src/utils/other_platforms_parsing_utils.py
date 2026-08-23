"""Parsing helpers for Claude, Grok, and DeepSeek exports, used by
src/web_search_decision/data_extraction_other_cai.py.

Each platform's raw export JSON has a different shape (ChatGPT/DeepSeek:
parent/children message trees; Claude: a flat chronological array; Grok: a
session with a `responses` list), so every function here takes an explicit
`platform` argument and dispatches to a platform-specific implementation.
The parallel, ChatGPT-only module is
src/utils/chatgpt_conversation_utils.py.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.common_io import *


def normalize_timestamp(ts, platform):
    """
    Convert a platform-native timestamp to a naive datetime.

    - chatgpt: Unix epoch (seconds or milliseconds)
    - claude:  ISO 8601 string (e.g., "2025-10-14T11:13:34.610305Z")
    - grok:    BSON dict (e.g., {"$date": {"$numberLong": "1775273426443"}})
    """
    if platform == "chatgpt":
        ts = float(ts)
        if ts > 1e12:  # milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts)

    if platform in ("claude", "deepseek"):
        # ISO 8601, with either "Z" (claude) or explicit offset like "+08:00" (deepseek)
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).replace(tzinfo=None)

    if platform == "grok":
        ms = int(ts["$date"]["$numberLong"])
        return datetime.fromtimestamp(ms / 1000)

    raise ValueError(
        f"Unknown platform: {platform!r}. "
        "Use 'chatgpt', 'claude', 'grok', or 'deepseek'."
    )


def sort_conversation(conversation, platform):
    """
    Sort a conversation into chronological message order.

    Args:
        conversation: The conversation data.
            - For "chatgpt": pass either the full conversation dict (with "mapping")
              or the mapping dict directly.
            - For "claude": pass either the full conversation dict (with "chat_messages")
              or the chat_messages list directly.
        platform: "chatgpt", "claude", "grok", or "deepseek"

    Returns: list of dicts with normalized fields:
        - role: "user" | "assistant" | "system" | "tool" (chatgpt)
                or "human" | "assistant" (claude)
        - text: str
        - timestamp: Unix epoch float (chatgpt) or ISO 8601 string (claude)
        - raw: original node/message object
    """
    if platform == "chatgpt":
        mapping = conversation["mapping"] if "mapping" in conversation else conversation
        return _sort_chatgpt(mapping)
    elif platform == "claude":
        messages = (
            conversation["chat_messages"]
            if isinstance(conversation, dict) and "chat_messages" in conversation
            else conversation
        )
        return _sort_claude(messages)
    elif platform == "grok":
        return _sort_grok(conversation)
    elif platform == "deepseek":
        return _sort_deepseek(conversation)
    else:
        raise ValueError(
            f"Unknown platform: {platform!r}. "
            "Use 'chatgpt', 'claude', 'grok', or 'deepseek'."
        )


def _sort_chatgpt(mapping):
    """ChatGPT: tree structure with parent/children pointers."""
    # Find root (parent is None or not in mapping), and derive parent -> children
    # links from the "parent" pointers as a fallback for exports where the
    # "children" field is absent or null on every node.
    root_id = None
    children_map = {}
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
        else:
            children_map.setdefault(parent, []).append(node_id)
    if root_id is None:
        raise ValueError("No root node found")

    ordered = []

    def dfs(node_id):
        node = mapping.get(node_id)
        if not node:
            return
        message = node.get("message")
        if message:
            author = message.get("author", {}) or {}
            content = message.get("content", {}) or {}
            parts = content.get("parts", []) or []
            text = "\n".join(str(p) for p in parts if p)
            ordered.append({
                "role": author.get("role"),
                "text": text,
                "timestamp": message.get("create_time"),
                "raw": node,
            })
        children = node.get("children") or children_map.get(node_id, [])
        for child_id in children:
            dfs(child_id)
            break  # follow only the first child (original behavior)

    dfs(root_id)
    return ordered


def _sort_claude(chat_messages):
    """Claude: flat array, already in order, but normalize the shape."""
    ordered = []
    for msg in chat_messages:
        content_blocks = msg.get("content") or []
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text" and block.get("text")
        ]
        text = "\n".join(text_parts) if text_parts else msg.get("text", "")

        ordered.append({
            "role": msg.get("sender"),
            "text": text,
            "timestamp": msg.get("created_at"),
            "raw": msg,
        })
    return ordered


def _grok_ts_ms(ts):
    """Extract Unix ms from Grok's BSON-style create_time, or 0 on miss."""
    if isinstance(ts, dict):
        inner = ts.get("$date")
        if isinstance(inner, dict):
            try:
                return int(inner.get("$numberLong"))
            except (TypeError, ValueError):
                return 0
    return 0


def _sort_grok(session):
    """Grok: session has `responses` (list of {"response": {...}}). Sort by create_time."""
    responses = session.get("responses") or []

    def ts_key(resp_wrap):
        resp = resp_wrap.get("response") or {}
        return _grok_ts_ms(resp.get("create_time"))

    ordered = []
    for resp_wrap in sorted(responses, key=ts_key):
        msg = resp_wrap.get("response") or {}
        text = msg.get("message", "") or ""
        ordered.append({
            "role": msg.get("sender"),
            "text": text,
            "timestamp": msg.get("create_time"),
            "raw": msg,
        })
    return ordered


def _sort_deepseek(conversation):
    """DeepSeek: ChatGPT-style mapping tree; root's parent is the literal 'root' sentinel."""
    mapping = conversation.get("mapping") or {}

    # Find root (parent is None, "root", or not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent == "root" or parent not in mapping:
            root_id = node_id
            break
    if root_id is None:
        return []

    ordered = []

    def dfs(node_id):
        node = mapping.get(node_id)
        if not node:
            return
        message = node.get("message")
        if message:
            ordered.append({
                "role": None,  # role inferred from fragment types in loader
                "text": "",
                "timestamp": message.get("inserted_at"),
                "raw": node,
            })
        for child_id in node.get("children", []):
            dfs(child_id)
            break  # follow only the first child (regen branches ignored)

    dfs(root_id)
    return ordered


def _resolve_topic_from_record(record):
    """Apply topic resolution policy for JSONL topic-label records.

    1. No labels (length 0) -> "Other".
    2. Single label -> use it.
    3. Multiple labels -> pick the topic with the highest frequency across
       ``data_per_turn[]``. On tie, the topic that appears first in turn
       order wins. Fall back to ``conversation_label[0]`` if turns are empty.
    """
    labels = record.get("conversation_label") or []
    if not labels:
        return "Other"
    if len(labels) == 1:
        return labels[0]

    turns = record.get("data_per_turn") or []
    freq = {}
    first_order = []
    for turn in turns:
        topic = turn.get("topic")
        if not topic:
            continue
        if topic in freq:
            freq[topic] += 1
        else:
            freq[topic] = 1
            first_order.append(topic)

    if not freq:
        # data_per_turn missing or empty: fall back to first conversation label.
        return labels[0]

    max_freq = max(freq.values())
    # Tie-breaker: the topic that appeared earliest in turn order.
    for topic in first_order:
        if freq[topic] == max_freq:
            return topic
    return labels[0]


def load_topics(platform):
    """Return a ``{conv_id: topic_string}`` lookup for the given platform.

    - chatgpt: external CSV with columns (conv_id, topic_new); NaN -> "Other".
    - claude / grok / deepseek: local JSONL under ``src/topics/``; resolution
      handled by ``_resolve_topic_from_record``.

    Neither source ships with this repo (both are research artifacts built
    over the paper's own ERB-restricted donated dataset), so in practice
    this returns {} for every platform on a fresh checkout -- by design, not
    as an error: callers must already treat a missing topic lookup as
    expected and fall back to a generic label (see
    src/web_search_decision/data_extraction_other_cai.py's
    topic-classification fallback for how a real topic gets assigned
    instead, from the conversation content itself).
    """
    if platform == "chatgpt":
        path = f"{OUTPUT_PATH}/chatgpt/metadata/All_Conversations_annotation.csv"
        try:
            df = pd.read_csv(path)[["conv_id", "topic_new"]]
        except Exception:
            return {}
        lookup = {}
        for _, row in df.iterrows():
            conv_id = row["conv_id"]
            topic = row["topic_new"]
            if pd.isna(topic) or str(topic).strip().lower() == "nan":
                lookup[conv_id] = "Other"
            else:
                lookup[conv_id] = str(topic)
        return lookup

    path = (
        Path(__file__).resolve().parent.parent
        / "topics" / f"{platform}_topic_labels.jsonl"
    )
    lookup = {}
    try:
        f = open(path)
    except OSError:
        return {}
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv_id = record.get("conversation_id")
            if not conv_id:
                continue
            lookup[conv_id] = _resolve_topic_from_record(record)
    return lookup
