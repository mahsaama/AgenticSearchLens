from datetime import datetime
import pandas as pd
from pathlib import Path
from src.utils.common_io import *


def normalize_timestamp(ts):
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
    topic_mapping_path = (
        f"{OUTPUT_PATH}/metadata/All_Conversations_annotation.csv"
    )
    try:
        topic_mapping_df = pd.read_csv(topic_mapping_path)[["conv_id", "topic_new"]]
        topic_lookup = dict(zip(topic_mapping_df["conv_id"], topic_mapping_df["topic_new"]))
    except:
        return dict()
    return topic_lookup


def load_turn_msgs(df, user_id, conv_id, turn_id, full=True):
    turn_msgs = (
        df[
            (df["user id"] == user_id)
            & (df["conv id"] == conv_id)
            & (df["turn id"] == turn_id)
        ]
        .reset_index()
        .loc[0, "turn_msgs"]
    )

    if full:
        return turn_msgs

    shortened_msgs = []
    for turn_msg in turn_msgs:
        msg = {
            "Sender": turn_msg.get("author", {}).get("role", ""),
            "Recipient": turn_msg.get("recipient", ""),
            "Message": turn_msg.get("content", {}),
        }
        shortened_msgs.append(msg)

    return shortened_msgs


def load_conv_msgs(user_id, conv_idx, sorted=True, full=True):
    file_path = Path(f"{DATA_BASE_PATH}/prolific_all_files/user_{user_id}/conversations.json")
    with open(file_path, "r") as f:
        data = json.load(f)

    conv = data[conv_idx]
    mapping = conv["mapping"]
    if sorted:
        mapping = sort_conversation(mapping)

    if full:
        return conv["title"], mapping
    
    shortened_mapping = []
    for mapping_msg in mapping:
        msg = {
            "Sender": mapping_msg.get("message").get("author", {}).get("role", ''),
            "Recipient": mapping_msg.get("message").get("recipient", ''),
            "Message": mapping_msg.get("message").get("content", {}),
        }
        shortened_mapping.append(msg)

    return conv["title"], shortened_mapping
