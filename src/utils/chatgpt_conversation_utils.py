"""Parsing helpers specific to ChatGPT's export format, used by
src/web_search_decision/data_extraction.py.

The parallel module for Claude/Grok/DeepSeek exports (whose raw JSON shapes
differ from ChatGPT's) is src/utils/other_platforms_parsing_utils.py.
"""

from datetime import datetime

import pandas as pd

from src.utils.common_io import *


def normalize_timestamp(ts):
    """Convert a ChatGPT `create_time` (Unix epoch, seconds or
    milliseconds) into a naive local datetime."""
    ts = float(ts)
    if ts > 1e12:  # milliseconds
        ts /= 1000
    return datetime.fromtimestamp(ts)


def sort_conversation(mapping):
    """
    Sort ChatGPT conversation mapping into conversational order,
    automatically detecting the root node.
    """

    # 1. Find root node (parent is None or missing), and derive parent -> children
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
            ordered.append(node)

        children = node.get("children") or children_map.get(node_id, [])
        for child_id in children:
            dfs(child_id)
            break

    dfs(root_id)
    return ordered


def load_topics():
    """Load the conv_id -> topic lookup from the paper's LLM-annotated
    conversation-topic CSV, if present.

    This CSV is a research artifact produced separately over the paper's own
    (ERB-restricted, unshared) donated dataset -- it will not exist for
    anyone running this pipeline on their own export. Callers must treat a
    missing file as expected, not an error: this returns an empty dict in
    that case, and data_extraction.py falls back to a topic of "Other" for
    every conversation when the lookup has no entry for it.
    """
    topic_mapping_path = (
        f"{OUTPUT_PATH}/chatgpt/metadata/All_Conversations_annotation.csv"
    )
    try:
        topic_mapping_df = pd.read_csv(topic_mapping_path)[["conv_id", "topic_new"]]
        topic_lookup = dict(zip(topic_mapping_df["conv_id"], topic_mapping_df["topic_new"]))
    except Exception:
        return dict()
    return topic_lookup
