"""Parses raw ChatGPT export files into a per-turn summary dataframe.

Reads data/chatgpt/user_<i>/conversations.json for i in range(NUM_USERS)
(see README.md's "Exporting Your Own Data") and, for every turn (one user
message + the assistant's response to it), extracts which tools were
called, reasoning/thinking-mode flags, message history, timing, and a
topic label -- writing the result to outputs/metadata/data_summary.* and
outputs/metadata/web_data_summary.* (parquet/pickle/csv).

Run directly as `python -m src.web_search_decision.chatgpt_extraction`.
The parallel module for Claude/Grok/DeepSeek exports is
src.web_search_decision.other_platforms_extraction.
"""

import json
import sys

sys.setrecursionlimit(5000)
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.chatgpt_conversation_utils import (
    DATA_BASE_PATH,
    OUTPUT_PATH,
    load_topics,
    normalize_timestamp,
    sort_conversation,
)
from src.utils.common_io import load_json
from src.utils.topic_classifier import classify_topic

try:
    from langdetect import detect
except ImportError:
    def detect(_text):
        return ""

NUM_USERS = 1


def load_whole_data():
    """Parse every user's conversations.json into a list of per-turn rows.

    Returns (all_data, num_conversations, num_turns, num_msgs, num_tool_usage),
    where all_data is a list of rows matching main()'s DataFrame columns.
    """
    all_data = []
    num_conversations = 0
    num_turns = 0
    num_msgs = 0
    num_tool_usage = 0
    tools = []
    topic_lookup = load_topics()

    def emit_turn(
        user_id, conv, turn_msgs, conv_starter, user_msg_history, assistant_msg_history
    ):
        """Build one output row (see main()'s DataFrame columns) from the
        messages making up a single user-prompt-to-assistant-response turn."""
        nonlocal num_tool_usage

        main_tool_calls = []
        reasoning_path = []
        thinking_path = []
        openai_models = []
        interactions = []
        thoughts = ""
        for turn_msg in turn_msgs:
            role = turn_msg.get("author", {}).get("name", "")
            if not role:
                role = turn_msg.get("author", {}).get("role", "")

            recipient = turn_msg.get("recipient", "all")
            ts = turn_msg.get("create_time")
            metadata = turn_msg.get("metadata", {})
            content = turn_msg.get("content", {})
            openai_models.append(metadata.get("model_slug", None))
            reasoning_path.append(metadata.get("reasoning_status", None))
            thinking_type = content.get("content_type", None)
            thinking_thoughts = content.get("thoughts", [])
            thinking_path.append(thinking_type)
            if "thoughts" == thinking_type:
                for tt in thinking_thoughts:
                    thoughts += tt.get("content", "")
                    thoughts += "\n"

            interactions.append(f"{role}:{recipient}")
            if ts and role == "assistant" and recipient != "all":
                main_tool_calls.append(recipient)
                num_tool_usage += 1

        reasoning = any(reasoning_path)
        thinking = "thoughts" in thinking_path

        time_ = pd.NaT
        for turn_msg in reversed(turn_msgs):
            ts = turn_msg.get("create_time")
            if ts is None:
                continue
            try:
                time_ = normalize_timestamp(ts)
                break
            except (TypeError, ValueError):
                continue

        topic_label = topic_lookup.get(conv["id"])
        if topic_label is None or str(topic_label).strip().lower() == "nan":
            # No annotation available for this conversation (the common
            # case outside the paper's own dataset) -- classify it from its
            # opening message instead of defaulting to "Other".
            first_user_message = user_msg_history[0] if user_msg_history else ""
            topic = classify_topic(first_user_message)
        else:
            topic = str(topic_label)

        try:
            language = detect("\n".join([str(x) for x in user_msg_history]))
        except Exception:
            language = ""

        tools.extend(main_tool_calls)
        return [
            user_id,
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
            openai_models,
            user_msg_history.copy(),
            assistant_msg_history.copy(),
            json.dumps(turn_msgs),
            time_,
        ]

    # Load all files
    for i in tqdm(range(NUM_USERS)):
        file_path = Path(
            f"{DATA_BASE_PATH}/user_{i}/conversations.json"
        )
        if not file_path.exists():
            continue

        data = load_json(file_path)

        for conv in data:
            num_conversations += 1
            mapping = sort_conversation(conv["mapping"])
            user_msg_history = []
            assistant_msg_history = []

            turn_msgs = []
            conv_starter = 1
            turn_has_user = False
            turn_has_assistant_response = False
            for msg_info in mapping:
                msg = msg_info.get("message")
                if not msg:
                    continue

                role = msg.get("author", {}).get("role", "")
                if role == "system":
                    continue

                if role == "user":
                    if turn_has_user and turn_has_assistant_response:
                        num_turns += 1
                        all_data.append(
                            emit_turn(
                                i,
                                conv,
                                turn_msgs,
                                conv_starter,
                                user_msg_history,
                                assistant_msg_history,
                            )
                        )
                        conv_starter = 0
                        num_msgs += len(turn_msgs)

                    turn_msgs = [msg]
                    turn_has_user = True
                    turn_has_assistant_response = False

                    temp = msg.get("content", {}).get("parts", [])
                    user_msg_history += [str(x) for x in temp]
                    continue
                elif turn_has_user:
                    turn_msgs.append(msg)
                else:
                    continue

                if role == "assistant" and msg.get("recipient", "all") == "all":
                    temp = msg.get("content", {}).get("parts", [])
                    assistant_msg_history += [str(x) for x in temp]
                    turn_has_assistant_response = True

            if turn_has_user and turn_has_assistant_response:
                num_turns += 1
                all_data.append(
                    emit_turn(
                        i,
                        conv,
                        turn_msgs,
                        conv_starter,
                        user_msg_history,
                        assistant_msg_history,
                    )
                )
                num_msgs += len(turn_msgs)

    return all_data, num_conversations, num_turns, num_msgs, num_tool_usage


def load_whole_data_from_file(fmt, base_dir=None):
    """Load a previously extracted data_summary.<fmt> back into a DataFrame."""
    metadata_dir = Path(base_dir) if base_dir else Path(f"{OUTPUT_PATH}/metadata")
    if fmt == "csv":
        df = pd.read_csv(metadata_dir / "data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(metadata_dir / "data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(metadata_dir / "data_summary.parquet")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return df


def load_web_data_from_file(fmt, base_dir=None):
    """Load a previously extracted web_data_summary.<fmt> back into a DataFrame
    (data_summary.* filtered down to turns that invoked a Web-search tool)."""
    metadata_dir = Path(base_dir) if base_dir else Path(f"{OUTPUT_PATH}/metadata")
    if fmt == "csv":
        df = pd.read_csv(metadata_dir / "web_data_summary.csv")
    elif fmt == "pkl":
        df = pd.read_pickle(metadata_dir / "web_data_summary.pkl")
    elif fmt == "parquet":
        df = pd.read_parquet(metadata_dir / "web_data_summary.parquet")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return df


def main():
    """Extract all ChatGPT conversations and write data_summary.* /
    web_data_summary.* under outputs/metadata/."""
    Path(f"{OUTPUT_PATH}/metadata").mkdir(parents=True, exist_ok=True)
    all_data, num_conversations, num_turns, num_msgs, num_tool_usage = load_whole_data()
    # Convert to DataFrame
    df = pd.DataFrame(
        all_data,
        columns=[
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
        ],
    )
    df = df.sort_values("time")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

    print(f"Number of Users: {NUM_USERS}")
    print(f"Number of Conversation: {num_conversations}")
    print(f"Number of Turns: {num_turns}")
    print(f"Number of Turns with Tool calls: {len(df[df['tools'].apply(len) > 0])}")
    print(f"Number of Messages: {num_msgs}")
    print(f"Number of Messages with Tool calls: {num_tool_usage}")

    df = df.reset_index(drop=True)
    df.to_parquet(f"{OUTPUT_PATH}/metadata/data_summary.parquet")
    df.to_pickle(f"{OUTPUT_PATH}/metadata/data_summary.pkl")
    df.to_csv(f"{OUTPUT_PATH}/metadata/data_summary.csv", index=False)
    print(f"All Data ({len(df)}) Saved Successfully!")

    web_df = df[(df["interactions"].apply(lambda x: "web" in str(x)))]

    web_df = web_df.reset_index(drop=True)
    web_df.to_parquet(f"{OUTPUT_PATH}/metadata/web_data_summary.parquet")
    web_df.to_pickle(f"{OUTPUT_PATH}/metadata/web_data_summary.pkl")
    web_df.to_csv(f"{OUTPUT_PATH}/metadata/web_data_summary.csv", index=False)
    print(f"Web Data ({len(web_df)}) Saved Successfully!")


if __name__ == "__main__":
    main()
