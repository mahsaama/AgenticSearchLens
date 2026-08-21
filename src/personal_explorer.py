"""Single-user, single-upload statistics for the "Explore Your Own Traces" page.

Unlike web_tool_invocation.py / query_reformulations.py / source_selection.py
(built for the paper's full donated cohort — cross-platform comparisons,
longitudinal trends across hundreds of users, LLM-judge evaluations), the
functions here compute a handful of general stats from exactly one person's
own extracted conversation history, for any of the four platforms uniformly.
They call straight into data_extraction.py / data_extraction_other_cai.py's
extraction logic in-memory and never write to outputs/ — nothing here
persists beyond the request.

Deliberately out of scope for now (see README/PR discussion): topic
breakdowns (need the paper's separately LLM-annotated topic CSV, not
derivable from a raw export) and retrieved/cited-domain analysis (today's
URL/citation parsing in source_selection.py is ChatGPT-only).
"""

from collections import Counter
from pathlib import Path

import pandas as pd

import src.data_extraction as chatgpt_extraction
import src.data_extraction_other_cai as other_extraction

PLATFORMS = ("chatgpt", "claude", "grok", "deepseek")


class NoDataError(ValueError):
    """Raised when a platform has no extractable conversations."""


def _dataframe_for_platform(platform, base_dir="data"):
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform!r}. Use one of {PLATFORMS}.")

    if platform == "chatgpt":
        chatgpt_extraction.DATA_BASE_PATH = f"{base_dir}/chatgpt"
        all_data, _num_conversations, _num_turns, _num_msgs, _num_tool_usage = (
            chatgpt_extraction.load_whole_data()
        )
        columns = [
            "user_id", "conv_id", "turn_id", "conv_starter", "topic", "language",
            "tools", "interactions", "reasoning", "thinking", "thoughts",
            "openai_models", "user_msg_history", "assistant_msg_history",
            "turn_msgs", "time",
        ]
    else:
        # load_whole_data() reads DATA_BASE_PATH as a plain module global that
        # main() normally sets via argparse; set it the same way main() would
        # so this is safely callable without going through the CLI.
        other_extraction.DATA_BASE_PATH = Path(base_dir) / platform
        (
            all_data, _num_users, _num_conversations, _num_turns,
            _num_msgs, _num_tool_usage,
        ) = other_extraction.load_whole_data(platform)
        columns = other_extraction.COLUMNS

    if not all_data:
        raise NoDataError(
            f"No conversations found for platform={platform!r}. "
            f"Expected {base_dir}/{platform}/user_0/conversations.json."
        )

    df = pd.DataFrame(all_data, columns=columns)
    df = df.sort_values("time").reset_index(drop=True)
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    return df


def explore_platform(platform, base_dir="data", top_n_tools=10):
    """Compute general, single-user stats for one platform's upload.

    Returns a JSON-safe dict: conversation/turn counts, Web-search rate
    (overall and by month), tool-usage counts, and reasoning-mode rate.
    """
    df = _dataframe_for_platform(platform, base_dir=base_dir)
    web_mask = other_extraction.web_call_mask(df, platform)

    total_turns = len(df)
    total_conversations = int(df["conv_id"].nunique())
    web_call_count = int(web_mask.sum())
    reasoning_count = int(pd.to_numeric(df["reasoning"], errors="coerce").fillna(0).sum())

    monthly = (
        pd.DataFrame({"month": df["month"], "is_web": web_mask})
        .groupby("month")
        .agg(turns=("is_web", "size"), web_turns=("is_web", "sum"))
        .reset_index()
        .sort_values("month")
    )
    monthly_trend = [
        {
            "month": row.month.strftime("%Y-%m"),
            "turns": int(row.turns),
            "web_turns": int(row.web_turns),
            "web_rate": round(100 * row.web_turns / row.turns, 1) if row.turns else 0.0,
        }
        for row in monthly.itertuples()
    ]

    tool_counts = Counter()
    for tools in df["tools"]:
        for tool in tools or []:
            tool_counts[str(tool)] += 1
    tool_usage = [
        {"tool": tool, "count": count}
        for tool, count in tool_counts.most_common(top_n_tools)
    ]

    return {
        "platform": platform,
        "total_conversations": total_conversations,
        "total_turns": total_turns,
        "web_call_count": web_call_count,
        "web_call_rate": round(100 * web_call_count / total_turns, 1) if total_turns else 0.0,
        "reasoning_rate": round(100 * reasoning_count / total_turns, 1) if total_turns else 0.0,
        "monthly_trend": monthly_trend,
        "tool_usage": tool_usage,
    }
