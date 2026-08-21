import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import sys
sys.setrecursionlimit(5000)


DATA_BASE_PATH = "../data"
OUTPUT_PATH = "../outputs"


def normalize_timestamp(ts):
    ts = float(ts)
    if ts > 1e12:  # milliseconds
        ts /= 1000
    return datetime.fromtimestamp(ts)


def load_json(file_path):
    """Helper function to load JSON files."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def load_topics():
    topic_mapping_path = (
        f"{OUTPUT_PATH}/metadata/All_Conversations_annotation.csv"
    )
    topic_mapping_df = pd.read_csv(topic_mapping_path)[["conv_id", "topic_new"]]
    topic_lookup = dict(zip(topic_mapping_df["conv_id"], topic_mapping_df["topic_new"]))
    return topic_lookup


def sort_conversation(mapping):
    """
    Sort ChatGPT conversation mapping into conversational order,
    automatically detecting the root node.
    """

    # 1. Find root node (parent is None or missing)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

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

        for child_id in node.get("children", []):
            dfs(child_id)
            break

    dfs(root_id)
    return ordered


all_data = []
num_conversations = 0
num_turns = 0
num_msgs = 0
num_tool_usage = 0
num_users = 310
tools = []
topic_lookup = load_topics()

# Load all files
for i in tqdm(range(num_users)):
    file_path = Path(f"{DATA_BASE_PATH}/prolific_all_files/user_{i}/conversations.json")
    if not file_path.exists():
        continue

    with open(file_path, "r") as f:
        data = json.load(f)

    for conv_idx, conv in enumerate(data):
        num_conversations += 1
        mapping = sort_conversation(conv["mapping"])

        turn_msgs = []
        for msg_info in mapping:
            msg = msg_info.get("message")
            if not msg:
                # print(i, msg_id, msg_info)
                continue

            role = msg.get("author", {}).get("role", "")
            if role == "system":
                continue

            turn_msgs.append(msg)
            end_turn = msg.get("end_turn")

            if end_turn:
                num_turns += 1
                # eval turn msgs
                main_tool_calls = []
                reasoning_path = []
                thinking_path = []
                openai_models = []
                interactions = []
                for turn_msg in turn_msgs:
                    role = turn_msg.get("author", {}).get("name", "")
                    if not role:
                        role = turn_msg.get("author", {}).get("role", "")
                    recipient = turn_msg.get("recipient")
                    ts = turn_msg.get("create_time")
                    openai_models.append(turn_msg["metadata"].get("model_slug", None))
                    reasoning_path.append(
                        turn_msg["metadata"].get("reasoning_status", None)
                    )
                    thinking_path.append(turn_msg["content"].get("content_type", None))
                    interactions.append(f"{role}:{recipient}")
                    if ts and role == "assistant" and recipient != "all":
                        main_tool_calls.append(recipient)
                        num_tool_usage += 1

                reasoning = False
                for path in reasoning_path:
                    if path:
                        reasoning = True
                        break
                time_ = normalize_timestamp(turn_msgs[-1].get("create_time"))
                topic_ = str(topic_lookup.get(conv["id"], "Other"))
                topic = topic_ if topic_ != "nan" else "Other"
                all_data.append(
                    [
                        i,
                        conv_idx,
                        conv["id"],
                        turn_msgs[-1]["id"],
                        topic,
                        main_tool_calls,
                        interactions,
                        reasoning_path,
                        int(reasoning),
                        thinking_path,
                        openai_models,
                        time_,
                        turn_msgs,
                    ]
                )
                tools += main_tool_calls

                # empty turn msgs for new one
                num_msgs += len(turn_msgs)
                turn_msgs = []


# Convert to DataFrame
df = pd.DataFrame(
    all_data,
    columns=[
        "user id",
        "conv_idx",
        "conv id",
        "turn id",
        "topic",
        "tools",
        "interactions",
        "reasoning_path",
        "reasoning",
        "thinking_path",
        "openai_models",
        "time",
        "turn_msgs",
    ],
)
df = df.sort_values("time")
df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
df.head()
df[
    [
        "user id",
        "conv_idx",
        "conv id",
        "turn id",
        "topic",
        "tools",
        "interactions",
        "reasoning_path",
        "reasoning",
        "thinking_path",
        "openai_models",
        "time",
    ]
].to_csv(f"{OUTPUT_PATH}/tool_summary.csv")
