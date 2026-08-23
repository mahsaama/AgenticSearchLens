"""§3 analyses: Web-search *decisions* -- when and how often agents call
Web search, by platform/model/topic/time, and whether harness instructions
vs. models' own judgment drive those calls (invitro developer-prompt swap
replays), plus whether calling search actually helped (invitro Web vs.
no-Web quality comparison, Prometheus/LLM-judge evaluations).

Same scope note as the other analysis modules: written for the paper's full
cohort, organized as a library of individually-runnable analysis functions
(see the __main__ call list), each writing its own figure/table under
outputs/web_tool_invocation/.
"""

import os
import ast
import json
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from src.utils.common_io import *
from src.utils.chatgpt_conversation_utils import *
from src.utils.figure_style import with_paper_style, styler
from src.web_search_decision.extraction import (
    DEEPSEEK_WEB_TOOLS as _CANONICAL_DEEPSEEK_WEB_TOOLS,
    GROK_WEB_TOOLS as _CANONICAL_GROK_WEB_TOOLS,
    _chatgpt_has_web_call,
    load_web_data_from_file,
    load_whole_data_from_file,
)
# extraction.load_whole_data_from_file/load_web_data_from_file already
# dispatch to any of the 4 platforms (default platform="chatgpt") since the
# ChatGPT-only and Claude/Grok/DeepSeek loaders were merged into one module.
# _chatgpt_has_web_call/_CANONICAL_*_WEB_TOOLS are the same web-call
# detection extraction.py's own web_call_mask() uses -- reused here instead
# of this file keeping its own, separately-maintained (and previously
# weaker/narrower) copies.


CONF = "./web_tool_invocation"

topic_to_situation_mapping = {
    "Travel": "High-Investment Recommendation",
    "Cars": "High-Investment Recommendation",
    "Mobile phones": "High-Investment Recommendation",
    "Gift Suggestion": "High-Investment Recommendation",
    "Fashion": "High-Investment Recommendation",
    "Household Work": "High-Investment Recommendation",
    "Event Planning": "High-Investment Recommendation",

    "Weather and Climate": "Volatile/Temporal Information",
    "Finance": "Volatile/Temporal Information",
    "Energy": "Volatile/Temporal Information",
    "Politics & History": "Volatile/Temporal Information",
    "Games": "Volatile/Temporal Information",
    "Music": "Volatile/Temporal Information",
    "Military": "Volatile/Temporal Information",

    "Health": "High-Stakes Accuracy",
    "Mental Health": "High-Stakes Accuracy",
    "Law": "High-Stakes Accuracy",
    "Security & Privacy": "High-Stakes Accuracy",

    "Scientific Writing": "External Reference",
    "Books": "External Reference",
    "Programming": "External Reference",

    "AI": "Low Confidence/Niche Fact",
    "GPT": "Low Confidence/Niche Fact",
    "Science": "Low Confidence/Niche Fact",
    "Physics": "Low Confidence/Niche Fact",
    "Astrology": "Low Confidence/Niche Fact",
    "Religion": "Low Confidence/Niche Fact",
    "Animals/Pets information": "Low Confidence/Niche Fact",
    "Art": "Low Confidence/Niche Fact",
    "Drinks": "Low Confidence/Niche Fact",

    "Languages": "Unfamiliar Term/Typo",
    "Time conversion": "Unfamiliar Term/Typo",

    "Math": "User Verification",
    "Troubleshooting": "User Verification",
    "Cooking": "User Verification",

    "Social Media Content Generation": "Attribution/Sourcing Needed",
    "Email Drafting": "Attribution/Sourcing Needed",
    "Job Search": "Attribution/Sourcing Needed",
    "Art Generation": "Attribution/Sourcing Needed",

    "Misc": "Explicit Command",
    "Roleplay": "Explicit Command"
}

def _primary_model(models):
    if not isinstance(models, list):
        return "Unknown"
    cleaned = [model for model in models if isinstance(model, str) and model]
    if not cleaned:
        return "Unknown"
    return cleaned[-1]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _is_missing_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_possible_literal(value):
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text[0] not in "[{":
        return value

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    return value


def _as_list_value(value):
    if _is_missing_value(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        try:
            return _as_list_value(value.tolist())
        except Exception:
            pass
    parsed = _parse_possible_literal(value)
    if parsed is not value:
        return _as_list_value(parsed)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [value]


def _stringify_nested_text(value):
    if _is_missing_value(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        parsed = _parse_possible_literal(text)
        if parsed is not text:
            return _stringify_nested_text(parsed)
        return text
    if isinstance(value, dict):
        for key in ("content", "text", "q"):
            if key in value:
                text = _stringify_nested_text(value.get(key))
                if text:
                    return text
        return "\n".join(
            text
            for text in (_stringify_nested_text(item) for item in value.values())
            if text
        ).strip()
    if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
        return "\n".join(
            text
            for text in (_stringify_nested_text(item) for item in value)
            if text
        ).strip()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _stringify_nested_text(value.tolist())
        except Exception:
            pass
    return str(value).strip()


def _parse_eval_json(text):
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                return {}
    return {}


def _trim_text(text, max_chars):
    text = str(text or "").strip()
    if not max_chars or len(text) <= max_chars:
        return text
    return text[-max_chars:].strip()


def _normalize_topic_name(topic):
    topic = str(topic or "").strip()
    if topic == "GPT":
        return "AI"
    return topic


def _safe_filename(value):
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ).strip("_")


def _json_safe_value(value):
    if _is_missing_value(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in value.items()}
    if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe_value(value.tolist())
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except Exception:
            pass
    return str(value)


def _metric_preference_counts(row, metric_names):
    prefer_none = 0
    prefer_required = 0
    ties = 0
    for metric in metric_names:
        none_score = pd.to_numeric(
            row.get(f"none_{metric}_score"), errors="coerce"
        )
        required_score = pd.to_numeric(
            row.get(f"required_{metric}_score"), errors="coerce"
        )
        if pd.isna(none_score) or pd.isna(required_score):
            continue
        if required_score > none_score:
            prefer_required += 1
        elif none_score > required_score:
            prefer_none += 1
        else:
            ties += 1

    return prefer_none, prefer_required, ties


def _strict_call_outcome(row, metric_names):
    prefer_none, prefer_required, _ = _metric_preference_counts(
        row, metric_names
    )
    total_metrics = len(metric_names)
    decision_uses_web = _parse_bool(row.get("decision_uses_web", False))

    # Right call if at least one selected metric agrees with the decision.
    if decision_uses_web:
        if prefer_required >= 1:
            return "Right Call"
        if prefer_none >= 1:
            return "Over Call"
        return "Right Call"

    if prefer_none >= 1:
        return "Right Call"
    if prefer_required == total_metrics and total_metrics > 0:
        return "Under Call"
    return "Right Call"


def _row_has_web_call(row):
    for col in ("tools", "interactions"):
        if col not in row:
            continue
        if any("web" in str(item).lower() for item in _as_list_value(row.get(col))):
            return True

    if "web_queries" in row:
        web_queries = _as_list_value(row.get("web_queries"))
        return bool(_stringify_nested_text(web_queries))

    return True


def _latest_user_message(row):
    user_msg_history = _as_list_value(row.get("user_msg_history", []))
    user_messages = [
        str(message).strip()
        for message in user_msg_history
        if str(message).strip()
    ]
    if user_messages:
        return user_messages[-1]

    for col in ("user_query", "prompt", "Prompt_with_history"):
        text = _stringify_nested_text(row.get(col, ""))
        if text:
            return text
    return ""


def _turn_messages(row):
    turn_msgs = row.get("turn_msgs", [])
    parsed = _parse_possible_literal(turn_msgs)
    return parsed if isinstance(parsed, list) else []


def _message_role(message):
    author = message.get("author", {}) if isinstance(message, dict) else {}
    return str(author.get("name") or author.get("role") or "").strip()


def _message_thought_text(message):
    if not isinstance(message, dict):
        return ""
    content = message.get("content", {})
    if not isinstance(content, dict):
        return ""
    if content.get("content_type") != "thoughts":
        return ""
    return _stringify_nested_text(content.get("thoughts", []))


def _message_has_web_call_or_query(message):
    if not isinstance(message, dict):
        return False

    role = _message_role(message)
    recipient = str(message.get("recipient", "")).lower()
    if role == "assistant" and recipient and recipient != "all" and "web" in recipient:
        return True

    metadata = message.get("metadata", {})
    if not isinstance(metadata, dict):
        return False

    if _as_list_value(metadata.get("search_queries", [])):
        return True

    search_model_queries = metadata.get("search_model_queries", {})
    if isinstance(search_model_queries, dict):
        if _as_list_value(search_model_queries.get("queries", [])):
            return True

    return False


def _thoughts_before_first_web_call(row):
    thoughts = []
    for message in _turn_messages(row):
        thought_text = _message_thought_text(message)
        if thought_text:
            thoughts.append(thought_text)

        if _message_has_web_call_or_query(message):
            break

    return "\n".join(thoughts).strip()


def _thoughts_before_first_web_query(row):
    web_queries = _as_list_value(row.get("web_queries", []))
    if not _stringify_nested_text(web_queries):
        return ""

    thoughts_list = _as_list_value(row.get("thoughts_list", []))
    if not thoughts_list:
        return ""

    return _stringify_nested_text(thoughts_list[0])


_WEB_TOOL_NAME_HINTS = ("web", "search", "browse", "browser")


def _auto_categorize_tool(tool_name):
    """Heuristic "Web & Browsing" vs "Plugins" classification for a single
    tool/recipient name, used only as a fallback when no real category file
    is present (see _tool_to_category_lookup). Reuses the same web-tool name
    sets extraction.py's web_call_mask() already relies on
    for Grok/DeepSeek, plus a generic name-substring check that covers
    ChatGPT/Claude's "web"-named tools."""
    name = str(tool_name or "").lower()
    if any(hint in name for hint in _WEB_TOOL_NAME_HINTS):
        return "Web & Browsing"
    if tool_name in _CANONICAL_GROK_WEB_TOOLS or tool_name in _CANONICAL_DEEPSEEK_WEB_TOOLS:
        return "Web & Browsing"
    return "Plugins"


class _AutoToolCategoryLookup(dict):
    """An empty dict that classifies any tool via _auto_categorize_tool
    instead of falling through to a caller-supplied default. Used in place
    of a real (but absent) category lookup so a missing
    all_tools_categorized.json degrades to a coarser Web-vs-other split
    rather than silently mapping every tool to "Plugins"/"Others" -- which
    would make every Web-search trend line in this file read 0%, a wrong
    answer delivered without any error."""

    def get(self, tool_name, default=None):
        return super().get(tool_name) or _auto_categorize_tool(tool_name)


def _tool_to_category_lookup(path):
    """Flat {tool_name: category} lookup from a categorized-tools JSON file
    (nested {category: {tool: id, ...}, ...}) at `path`, or an
    auto-classifying fallback (_AutoToolCategoryLookup) if the file isn't
    present -- true for anyone but the paper's authors, who built it by
    hand over their own dataset's observed tool names."""
    tools_categorized = load_json(path)
    if not tools_categorized:
        return _AutoToolCategoryLookup()
    return {
        tool: cat
        for cat, cat_tools in tools_categorized.items()
        for tool in cat_tools
    }


def web_call_trend_over_time(df):
    df = df.copy()
    tool_to_category = _tool_to_category_lookup(f"{OUTPUT_PATH}/chatgpt/metadata/all_tools_categorized.json")
    df["categories"] = df["tools"].apply(lambda x: [tool_to_category.get(t, "Plugins") for t in x])

    df["month"] = pd.to_datetime(df["month"])
    df["tool_used"] = df["tools"].apply(lambda x: isinstance(x, list) and len(x) > 0)

    monthly_tooly_turns = (
        df.groupby("month")["tool_used"]
        .mean()
        .reset_index(name="tooly_turns")
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_tooly_turns["month"],
        y=monthly_tooly_turns["tooly_turns"],
        mode="lines+markers",
        name="All tool calls",
    ))

    cat = "Web & Browsing"
    df["cat_used"] = df["categories"].apply(lambda x: isinstance(x, list) and cat in x)
    monthly_tooly_turns = (
        df.groupby("month")["cat_used"]
        .mean()
        .reset_index(name="cat_tooly_turns")
        .sort_values("month")
    )
    fig.add_trace(go.Scatter(
        x=monthly_tooly_turns["month"],
        y=monthly_tooly_turns["cat_tooly_turns"],
        mode="lines+markers",
        name=f"With web call",
    ))

    df["cat_used"] = df["categories"].apply(lambda x: isinstance(x, list) and len(x) > 0 and cat not in x)
    monthly_tooly_turns = (
        df.groupby("month")["cat_used"]
        .mean()
        .reset_index(name="cat_tooly_turns")
        .sort_values("month")
    )
    fig.add_trace(go.Scatter(
        x=monthly_tooly_turns["month"],
        y=monthly_tooly_turns["cat_tooly_turns"],
        mode="lines+markers",
        name=f"Without web call",
    ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Turns (%)",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "tooly_turns_rate_over_time"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 17))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def web_call_trend_over_time_all_convai(df):
    min_valid_month = pd.Timestamp("2023-01-01")
    cat = "Web & Browsing"

    def _monthly_web_rate(platform_df, tool_to_category):
        platform_df = platform_df.copy()
        platform_df["tools"] = platform_df["tools"].apply(_as_list_value)
        platform_df["categories"] = platform_df["tools"].apply(
            lambda tools: [tool_to_category.get(tool, "Others") for tool in tools]
        )
        platform_df["month"] = (
            pd.to_datetime(platform_df["month"], errors="coerce", utc=True)
            .dt.tz_convert(None)
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        platform_df = platform_df.dropna(subset=["month"])
        platform_df = platform_df[platform_df["month"] >= min_valid_month].copy()

        platform_df["cat_used"] = platform_df["categories"].apply(lambda cats: cat in cats)
        total_web_turns = int(platform_df["cat_used"].sum())
        monthly_rate = (
            platform_df.groupby("month")["cat_used"]
            .mean()
            .reset_index(name="cat_tooly_turns")
            .sort_values("month")
        )
        return monthly_rate, total_web_turns, len(platform_df)

    openai_tool_to_category = _tool_to_category_lookup(f"{OUTPUT_PATH}/chatgpt/metadata/all_tools_categorized.json")
    openai_monthly, openai_total_web_turns, openai_total_turns = _monthly_web_rate(
        df, openai_tool_to_category
    )
    overall_rates = [
        {
            "platform": "ChatGPT",
            "web_call_turns": openai_total_web_turns,
            "num_turns": openai_total_turns,
        }
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=openai_monthly["month"],
        y=openai_monthly["cat_tooly_turns"],
        mode="lines+markers",
        name=f"ChatGPT",
    ))

    for cai in ["claude", "grok", "deepseek"]:
        cai_df = load_whole_data_from_file(fmt="pkl", platform=cai)
        cai_tool_to_category = _tool_to_category_lookup(f"{OUTPUT_PATH}/{cai}/metadata/all_tools_categorized.json")
        monthly_tooly_turns, total_web_turns, total_turns = _monthly_web_rate(
            cai_df, cai_tool_to_category
        )
        overall_rates.append(
            {
                "platform": cai.capitalize(),
                "web_call_turns": total_web_turns,
                "num_turns": total_turns,
            }
        )
        fig.add_trace(go.Scatter(
            x=monthly_tooly_turns["month"],
            y=monthly_tooly_turns["cat_tooly_turns"],
            mode="lines+markers",
            name=cai.capitalize(),
        ))

    overall_rates_df = pd.DataFrame(overall_rates)
    print(overall_rates_df.to_string(index=False))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="% Chatbot Responses",
        xaxis=dict(
            type="date",
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90, t=30),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "tooly_turns_rate_over_time_across_convais"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(20, 24), legend_pos=(0.5, 1))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")



def web_call_trend_over_time_by_model(df):
    df = df.copy()
    selected_models = ['gpt-4-1', 'gpt-4-1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-instant', 'gpt-5-mini', 'gpt-5-thinking', 'gpt-5-2', 'gpt-5-2-thinking', 'o3', 'o3-mini', 'text-davinci-002-render-sha']
    tool_to_category = _tool_to_category_lookup(f"{OUTPUT_PATH}/chatgpt/metadata/all_tools_categorized.json")
    df["categories"] = df["tools"].apply(lambda x: [tool_to_category.get(t, "Plugins") for t in x])
    df["month"] = pd.to_datetime(df["month"])
    df["model"] = df["openai_models"].apply(_primary_model)
    df = df[df["month"] >= pd.Timestamp("2024-01-01")].copy()

    cat = "Web & Browsing"
    df["cat_used"] = df["categories"].apply(lambda x: isinstance(x, list) and cat in x)
    plot_df = (
        df.groupby(["month", "model"])["cat_used"]
        .mean()
        .reset_index(name="web_call_rate")
        .sort_values(["model", "month"])
    )
    plot_df = plot_df[plot_df["model"].str.lower() != "unknown"].copy()
    plot_df = plot_df[plot_df["model"].isin(selected_models)].copy()

    color_pool = (
        qualitative.Light24
        + qualitative.Set3
        + qualitative.Alphabet
        + qualitative.Dark24
    )

    fig = go.Figure()
    for idx, model in enumerate(selected_models):
        if model not in set(plot_df["model"].unique()):
            continue
        model_df = plot_df[plot_df["model"] == model]
        fig.add_trace(
            go.Scatter(
                x=model_df["month"],
                y=model_df["web_call_rate"],
                mode="lines+markers",
                name=model,
                line=dict(color=color_pool[idx % len(color_pool)]),
                marker=dict(color=color_pool[idx % len(color_pool)]),
            )
        )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Turns with Web Call (%)",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        # margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "web_call_trend_over_time_by_model"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 14), legend_pos=(0.8, 1.8))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def web_call_stacked_bar_by_platform_year(platform_configs=None):
    if platform_configs is None:
        platform_configs = [
            ("openai", "OpenAI"),
            ("claude", "Claude"),
            ("grok", "Grok"),
            ("deepseek", "DeepSeek"),
        ]
    year_map = {
        "openai": [2024, 2025],
        "claude": [2024, 2025],
        "grok": [2025, 2026],
        "deepseek": [2024, 2025],
    }

    def _load_platform_df(platform):
        # This file's "openai" convention maps to extraction.py's "chatgpt".
        return load_whole_data_from_file(
            fmt="pkl", platform="chatgpt" if platform == "openai" else platform
        )

    plot_frames = []
    for platform, display_name in platform_configs:
        try:
            df = _load_platform_df(platform).copy()
        except Exception as e:
            print(f"Failed to load data for `{platform}`: {e}")
            continue

        if "topic" not in df.columns:
            print(f"Skipping `{platform}` because `topic` column is missing.")
            continue
        if "month" not in df.columns:
            print(f"Skipping `{platform}` because `month` column is missing.")
            continue

        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df[df["month"].notna()].copy()
        df["year"] = df["month"].dt.year
        selected_years = year_map.get(platform, [2024, 2025])
        df = df[df["year"].isin(selected_years)].copy()
        df["topic"] = df["topic"].fillna("Other").apply(_normalize_topic_name)
        df = df[
            ~df["topic"].astype(str).str.lower().isin({"other", "uncategorized", "misc"})
        ].copy()
        if df.empty:
            year_text = "/".join(str(year) for year in selected_years)
            print(f"No platform-year-topic rows found for `{platform}` in {year_text}.")
            continue

        platform_df = (
            df.groupby(["year", "topic"])
            .size()
            .reset_index(name="count")
        )
        platform_df["platform"] = platform
        platform_df["platform_display"] = display_name
        platform_df["bar_label"] = platform_df["year"].apply(
            lambda year: f"{display_name} {year}"
        )
        plot_frames.append(platform_df)

    if not plot_frames:
        print("No platform-year-topic rows found for the requested platform/year pairs.")
        return pd.DataFrame()

    plot_df = pd.concat(plot_frames, ignore_index=True)
    plot_df["total_turns"] = (
        plot_df.groupby(["platform", "year"])["count"]
        .transform("sum")
    )
    plot_df["topic_share"] = plot_df["count"].div(
        plot_df["total_turns"].replace(0, pd.NA)
    ).fillna(0.0)

    topic_order = (
        plot_df.groupby("topic")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    other_topics = sorted(
        topic for topic in plot_df["topic"].unique() if topic not in topic_order
    )
    topic_order = topic_order + other_topics

    available_labels = set(plot_df["bar_label"])
    bar_order = []
    for platform, display_name in platform_configs:
        for year in year_map.get(platform, [2024, 2025]):
            label = f"{display_name} {year}"
            if label in available_labels:
                bar_order.append(label)

    bar_year_topic_tuples = []
    for platform, display_name in platform_configs:
        for year in year_map.get(platform, [2024, 2025]):
            label = f"{display_name} {year}"
            for topic in topic_order:
                bar_year_topic_tuples.append((display_name, year, label, topic))

    platform_year_index = pd.MultiIndex.from_tuples(
        bar_year_topic_tuples,
        names=["platform_display", "year", "bar_label", "topic"],
    )
    plot_df = (
        plot_df.set_index(["platform_display", "year", "bar_label", "topic"])
        .reindex(platform_year_index)
        .reset_index()
    )
    for col in ["count", "total_turns", "topic_share"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0)
    plot_df["total_turns"] = (
        plot_df.groupby(["bar_label"])["count"]
        .transform("sum")
    )
    plot_df["topic_share"] = plot_df["count"].div(
        plot_df["total_turns"].replace(0, pd.NA)
    ).fillna(0.0)

    plot_df["bar_label"] = pd.Categorical(
        plot_df["bar_label"], categories=bar_order, ordered=True
    )
    plot_df["topic"] = pd.Categorical(
        plot_df["topic"], categories=topic_order, ordered=True
    )
    plot_df = plot_df.sort_values(["topic", "bar_label"])

    fig = go.Figure()
    color_pool = (
        qualitative.Light24
        + qualitative.Set3
        + qualitative.Alphabet
        + qualitative.Dark24
    )
    legend_year = int(sorted(plot_df["year"].dropna().unique())[0])

    for idx, topic in enumerate(topic_order):
        topic_df = plot_df[plot_df["topic"] == topic].copy()
        if topic_df.empty:
            continue
        for year in sorted(topic_df["year"].dropna().unique()):
            year_df = topic_df[topic_df["year"] == year].copy()
            if year_df.empty or year_df["total_turns"].sum() == 0:
                continue
            fig.add_trace(
                go.Bar(
                    x=year_df["topic_share"],
                    y=year_df["bar_label"].astype(str),
                    name=topic,
                    legendgroup=str(topic),
                    orientation="h",
                    marker_color=color_pool[idx % len(color_pool)],
                    showlegend=bool(year == legend_year),
                    customdata=year_df[["year", "count", "total_turns"]],
                    hovertemplate=(
                        "Platform: %{y}<br>"
                        "Year: %{customdata[0]}<br>"
                        f"Topic: {topic}<br>"
                        "Share: %{x:.1%}<br>"
                        "Topic turns: %{customdata[1]}<br>"
                        "Total turns: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Topic share of all turns",
        yaxis_title="Platform",
        legend_title="",
        margin=dict(l=110, r=30, t=220, b=70),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(bar_order)),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="left",
            x=-0.2,
            traceorder="normal",
            font=dict(size=10),
            itemsizing="constant",
        ),
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    file_name = "web_call_stacked_bar_by_platform_year"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 14), legend_pos=(-0.2, 1.1))
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=16))
    fig.update_layout(
        margin=dict(l=110, r=30, t=220, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="left",
            x=-0.2,
            traceorder="normal",
            font=dict(size=10),
            itemsizing="constant",
        )
    )
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")
    return plot_df


def web_call_stacked_bar_by_model_year(df=None):
    return web_call_stacked_bar_by_platform_year()


def print_available_models(df):
    df = df.copy()
    df["model"] = df["openai_models"].apply(_primary_model)
    models = sorted(
        model
        for model in df["model"].dropna().unique()
        if isinstance(model, str) and model.lower() != "unknown"
    )
    print(models)
    return models


def topic_distribution_of_web_data(web_df):
    # in the turns that trigger web call, what is the percentage of each one: what are the important topics that had called web?
    # bar plot
    topic_counts = (
        web_df["topic"]
        .fillna("Other")
        .fillna("Misc")
        .apply(_normalize_topic_name)
        .loc[lambda x: x != "Other"]
        .value_counts(normalize=True)
        .rename_axis("topic")
        .reset_index(name="rate")
        .sort_values("rate", ascending=False)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=topic_counts["topic"],
            y=topic_counts["rate"],
            showlegend=False
        )
    )
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Share of web-call turns",
        xaxis=dict(
            tickangle=-45,
        ),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "topic_distribution_of_web_data"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 10))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

#      topic  rate                                                                                  
# 0                           Finance  6440                                                                                  
# 1                             Games  3919                                                                                  
# 2                            Health  3442                                                                                  
# 3                   Troubleshooting  2347                                                                                  
# 4                            Travel  2284                                                                                  
# 5                Politics & History  1502                                                                                  
# 6                              Cars  1446                                                                                  
# 7                          Roleplay  1427                                                                                  
# 8                     Mental Health  1278                                                                                  
# 9                           Fashion  1268                                                                                  
# 10                    Mobile phones  1158                                                                                  
# 11                            Music  1047                                                                                  
# 12                          Cooking   992                                                                                  
# 13                   Art Generation   898                                                                                  
# 14  Social Media Content Generation   877                                                                                  
# 15                        Languages   799                                                                                  
# 16                       Job Search   710
# 17                            Books   668
# 18                  Gift Suggestion   652
# 19                              Law   635
# 20                   Email Drafting   553
# 21         Animals/Pets information   529
# 22              Weather and Climate   465
# 23                   Household Work   448
# 24                      Programming   421
# 25                           Energy   413
# 26                          Science   385
# 27                         Religion   348
# 28                        Astrology   260
# 29                           Drinks   260
# 30                             Misc   244
# 31                             Math   237
# 32               Security & Privacy   216
# 33                              GPT   191
# 34             Gender and Diversity   190
# 35                         Military   143
# 36                  Time conversion   112
# 37                              Art    76
# 38               Scientific Writing    49
# 39                   Event Planning    30

 
def topic_distriction_of_whole_data(df):
    # in the whole turns, rate of each topic calling the web over number of all turns with that topic: what topics are more prune to call web?
    # bar plot. ChatGPT-only (no platform param): uses the `tools` column's
    # two-step web-call detection (see _chatgpt_has_web_call's docstring)
    # instead of a plain "web" substring match on `interactions`, which
    # missed legacy exports whose recipient was e.g. "browser".
    df["topic"] = df["topic"].fillna("Other").apply(_normalize_topic_name)
    df = df[(df["topic"] != "Other") & (df["topic"] != "Misc")].copy()
    df["has_web_call"] = df["tools"].apply(_chatgpt_has_web_call)
    all_topics = sorted(
        set(df["topic"].dropna().astype(str))
        | {
            _normalize_topic_name(topic)
            for topic in topic_to_situation_mapping
            if topic not in {"Other", "Misc"}
        }
    )

    topic_rates = (
        df.groupby("topic")["has_web_call"]
        .mean()
        .reindex(all_topics, fill_value=0.0)
        .rename("web_call_rate")
        .reset_index()
        .sort_values("web_call_rate", ascending=False)
    )
    print("topic rates:", topic_rates)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=topic_rates["topic"],
            y=topic_rates["web_call_rate"],
            showlegend=False
        )
    )
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Web-call rate over all turns",
        xaxis=dict(
            tickangle=-45,
        ),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = "topic_distribution_of_whole_data"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 10))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def topic_prompt_volume_and_web_rate_over_time(df, top_k=5):
    # ChatGPT-only (no platform param) -- see topic_distriction_of_whole_data's
    # comment on why `tools` + _chatgpt_has_web_call is used instead of a
    # "web" substring match on `interactions`.
    df = df.copy()
    output_dir = f"{OUTPUT_PATH}/{CONF}/topic_prompt_volume_and_web_rate_over_time"
    os.makedirs(output_dir, exist_ok=True)

    df["topic"] = df["topic"].fillna("Other").apply(_normalize_topic_name)
    df = df[
        ~df["topic"].astype(str).str.lower().isin({"other", "misc", "uncategorized"})
    ].copy()
    if df.empty:
        print("No topic rows available after filtering.")
        return pd.DataFrame()

    df["has_web_call"] = df["tools"].apply(_chatgpt_has_web_call)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df[df["month"].notna()].copy()
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    if df.empty:
        print("No valid month values found.")
        return pd.DataFrame()

    monthly_total_prompts = (
        df.groupby("month")
        .size()
        .rename("total_prompts")
    )

    topic_web_call_rate = (
        df.groupby("topic")["has_web_call"]
        .mean()
        .sort_values(ascending=False)
    )
    top_topics_whole = topic_web_call_rate.head(top_k).index.tolist()

    topic_web_share = (
        df[df["has_web_call"]]
        .groupby("topic")
        .size()
        .sort_values(ascending=False)
    )
    top_topics_web = topic_web_share.head(top_k).index.tolist()

    selected_topics = list(dict.fromkeys(top_topics_whole + top_topics_web))
    if not selected_topics:
        print("No top topics found for plotting.")
        return pd.DataFrame()

    month_index = pd.Index(sorted(df["month"].unique()), name="month")
    summary_rows = []
    color_pool = (
        qualitative.Plotly
        + qualitative.Light24
        + qualitative.Set3
        + qualitative.Alphabet
        + qualitative.Dark24
    )
    topic_colors = {
        topic: color_pool[idx % len(color_pool)]
        for idx, topic in enumerate(selected_topics)
    }

    for topic in selected_topics:
        topic_df = df[df["topic"] == topic].copy()
        if topic_df.empty:
            continue

        monthly = (
            topic_df.groupby("month")
            .agg(
                user_prompts=("topic", "size"),
                web_call_rate=("has_web_call", "mean"),
            )
            .reindex(month_index, fill_value=0)
            .reset_index()
        )
        monthly["total_prompts"] = monthly["month"].map(monthly_total_prompts).fillna(0)
        monthly["prompt_share"] = (
            monthly["user_prompts"]
            .div(monthly["total_prompts"].replace(0, pd.NA))
            .fillna(0.0)
        )
        monthly["prompt_share_percentage"] = monthly["prompt_share"] * 100.0
        monthly["web_call_percentage"] = monthly["web_call_rate"] * 100.0
        monthly["topic"] = topic
        summary_rows.append(monthly.copy())

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_df.to_csv(f"{output_dir}/topic_prompt_volume_and_web_rate_over_time.csv", index=False)
        to_json(
            {
                "top_topics_from_whole_data": top_topics_whole,
                "top_topics_from_web_data": top_topics_web,
                "selected_topics": selected_topics,
            },
            f"{output_dir}/selected_topics.json",
        )

        prompt_share_fig = go.Figure()
        web_rate_fig = go.Figure()
        for topic in selected_topics:
            topic_monthly = summary_df[summary_df["topic"] == topic].copy()
            if topic_monthly.empty:
                continue

            topic_monthly = topic_monthly.sort_values("month")
            color = topic_colors[topic]
            prompt_share_fig.add_trace(
                go.Scatter(
                    x=topic_monthly["month"],
                    y=topic_monthly["prompt_share_percentage"],
                    mode="lines+markers",
                    name=topic,
                    line=dict(color=color, width=3),
                    marker=dict(color=color, size=7),
                    hovertemplate=(
                        "Topic: " + topic + "<br>"
                        "Month: %{x|%b %Y}<br>"
                        "Prompt share: %{y:.1f}%<extra></extra>"
                    ),
                )
            )
            web_rate_fig.add_trace(
                go.Scatter(
                    x=topic_monthly["month"],
                    y=topic_monthly["web_call_percentage"],
                    mode="lines+markers",
                    name=topic,
                    line=dict(color=color, width=3),
                    marker=dict(color=color, size=7),
                    hovertemplate=(
                        "Topic: " + topic + "<br>"
                        "Month: %{x|%b %Y}<br>"
                        "Web-calling prompts: %{y:.1f}%<extra></extra>"
                    ),
                )
            )

        plot_specs = [
            (
                prompt_share_fig,
                "Topic Share of All User Prompts Over Time",
                "Prompt Share (%)",
                "topic_prompt_share_over_time",
            ),
            (
                web_rate_fig,
                "Web-Calling Prompt Rate by Topic Over Time",
                "Web-Calling Prompts (%)",
                "topic_web_call_rate_over_time",
            ),
        ]

        for fig, title, yaxis_title, file_name in plot_specs:
            fig.update_layout(
                title=title,
                xaxis_title="Month",
                yaxis_title=yaxis_title,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(l=70, r=40, t=80, b=90),
            )
            fig.update_xaxes(
                dtick="M2",
                tickformat="%b %Y",
                tickangle=-45,
            )
            fig.update_yaxes(
                tickformat=".0f",
                ticksuffix="%",
                rangemode="tozero",
            )
            fig.write_html(f"{output_dir}/{file_name}.html")

            paper_fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.5, 1.12))
            paper_fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.12,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(l=70, r=40, t=80, b=90),
            )
            try:
                paper_fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
            except Exception as e:
                first_error_line = str(e).strip().splitlines()[0] if str(e).strip() else e
                print(f"Could not write PDF for `{file_name}`; HTML was saved. {first_error_line}")
    return summary_df


def _has_web_call_for_platform(row, platform):
    """One turn's web-call detection, dispatched by platform. Was two
    independently-maintained implementations that had drifted out of sync
    with extraction.py's own detection (this file's ChatGPT branch used a
    "web" substring match on `interactions` that missed legacy exports
    whose recipient was e.g. "browser"; its Grok/DeepSeek tool-name sets
    were a separate, narrower copy of extraction.py's). Both are gone now
    -- this delegates to the exact same logic extraction.web_call_mask()
    uses, so results here can't silently disagree with what extraction
    itself would report as a web-call turn."""
    tools = _as_list_value(row.get("tools"))
    if platform in {"openai", "chatgpt"}:
        return _chatgpt_has_web_call(tools)
    if platform == "claude":
        return any("web" in str(tool).lower() for tool in tools)
    if platform == "grok":
        return any(str(tool) in _CANONICAL_GROK_WEB_TOOLS for tool in tools)
    if platform == "deepseek":
        return any(str(tool) in _CANONICAL_DEEPSEEK_WEB_TOOLS for tool in tools)
    return False


def topic_distriction_of_whole_data_all_platforms(platform_configs=None):
    if platform_configs is None:
        platform_configs = [
            ("openai", "OpenAI"),
            ("claude", "Claude"),
            ("grok", "Grok"),
            ("deepseek", "DeepSeek"),
        ]
    
    def _load_platform_df(platform):
        # This file's "openai" convention maps to extraction.py's "chatgpt".
        return load_whole_data_from_file(
            fmt="pkl", platform="chatgpt" if platform == "openai" else platform
        )

    platform_rate_frames = []
    for platform, display_name in platform_configs:
        try:
            df = _load_platform_df(platform).copy()
        except Exception as e:
            print(f"Failed to load data for `{platform}`: {e}")
            continue

        if "topic" not in df.columns:
            print(f"Skipping `{platform}` because `topic` column is missing.")
            continue

        if "tools" not in df.columns:
            print(f"Skipping `{platform}` because `tools` column is missing.")
            continue

        df["topic"] = (
            df["topic"]
            .fillna("Other")
            .apply(_normalize_topic_name)
        )
        df = df[
            ~df["topic"].str.lower().isin({"other", "uncategorized"})
        ].copy()
        if df.empty:
            print(f"No categorized topics found for `{platform}`.")
            continue

        df["has_web_call"] = df.apply(
            lambda row: _has_web_call_for_platform(row, platform),
            axis=1,
        )
        topic_rates = (
            df.groupby("topic")["has_web_call"]
            .mean()
            .reset_index(name="web_call_rate")
        )
        topic_rates["platform"] = platform
        topic_rates["platform_display"] = display_name
        platform_rate_frames.append(topic_rates)

    if not platform_rate_frames:
        print("No topic-rate rows found for any platform.")
        return pd.DataFrame()

    rates_df = pd.concat(platform_rate_frames, ignore_index=True)
    openai_rates = (
        rates_df[rates_df["platform"] == "openai"]
        .set_index("topic")["web_call_rate"]
    )
    openai_sorted_topics = openai_rates.sort_values(ascending=False).index.tolist()
    other_topics = sorted(
        topic for topic in rates_df["topic"].unique() if topic not in openai_rates
    )
    topic_order = openai_sorted_topics + other_topics

    platform_order = [
        display_name
        for platform, display_name in platform_configs
        if platform in set(rates_df["platform"])
    ]
    table_pct = (
        rates_df.pivot_table(
            index="topic",
            columns="platform_display",
            values="web_call_rate",
            aggfunc="mean",
        )
        .reindex(index=topic_order, columns=platform_order)
        .fillna(0.0)
        * 100.0
    )

    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
    file_name = "topic_distribution_of_whole_data_all_platforms"
    table_pct.to_csv(f"{OUTPUT_PATH}/chatgpt/metadata/{file_name}_percentages.csv")
    rates_df.to_csv(f"{OUTPUT_PATH}/chatgpt/metadata/{file_name}_long.csv", index=False)

    fig = go.Figure()
    for platform, display_name in platform_configs:
        platform_df = rates_df[rates_df["platform"] == platform].copy()
        if platform_df.empty:
            continue
        platform_df["topic"] = pd.Categorical(
            platform_df["topic"],
            categories=topic_order,
            ordered=True,
        )
        platform_df = platform_df.sort_values("topic")
        fig.add_trace(
            go.Bar(
                x=platform_df["topic"].astype(str),
                y=platform_df["web_call_rate"],
                name=display_name,
                customdata=(platform_df["web_call_rate"] * 100).round(2),
                hovertemplate=(
                    "Platform: %{fullData.name}<br>"
                    "Topic: %{x}<br>"
                    "Web-call rate: %{customdata:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Topic",
        yaxis_title="Web-call rate over all turns",
        xaxis=dict(
            categoryorder="array",
            categoryarray=topic_order,
            tickangle=-45,
        ),
        legend_title="",
        margin=dict(l=70, r=30, t=50, b=180),
    )
    fig.update_yaxes(tickformat=".0%")
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 14), legend_pos=(0.8, 1.25))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=14))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    print("\nTopic web-call rates by platform (%):")
    print(table_pct.round(2).to_string())
    return table_pct

def _load_tool_intent_input(input_path=None, input_fmt="parquet"):
    if input_path is None:
        try:
            return load_web_data_from_file(fmt=input_fmt).copy()
        except FileNotFoundError:
            for fallback_fmt in ("pkl", "csv", "parquet"):
                if fallback_fmt == input_fmt:
                    continue
                try:
                    return load_web_data_from_file(fmt=fallback_fmt).copy()
                except FileNotFoundError:
                    continue
            raise

    ext = os.path.splitext(str(input_path))[1].lower()
    if ext == ".csv":
        return pd.read_csv(input_path)
    if ext == ".pkl":
        return pd.read_pickle(input_path)
    if ext == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input file type: {ext}")


def _normalise_output_base(output_base, model_name):
    if output_base is None:
        safe_model_name = str(model_name).replace("/", "_")
        output_base = (
            f"{OUTPUT_PATH}/chatgpt/metadata/web_call_tool_intent_from_thoughts_"
            f"{safe_model_name}"
        )

    root, ext = os.path.splitext(str(output_base))
    if ext in {".csv", ".pkl", ".json"}:
        return root
    return str(output_base)


def _tool_intent_record_key(record):
    conv_id = str(record.get("conv_id", "")).strip()
    turn_id = str(record.get("turn_id", "")).strip()
    source_index = str(record.get("source_index", "")).strip()
    if conv_id or turn_id:
        return (conv_id, turn_id, source_index)
    return ("", "", source_index)


def _load_existing_tool_intent_records(output_base):
    csv_path = f"{output_base}.csv"
    if not os.path.exists(csv_path):
        return [], set()

    existing_df = pd.read_csv(csv_path)
    records = existing_df.to_dict(orient="records")
    keys = {_tool_intent_record_key(record) for record in records}
    return records, keys


def _save_tool_intent_records(records, output_base):
    output_dir = os.path.dirname(output_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    safe_records = [
        {key: _json_safe_value(value) for key, value in record.items()}
        for record in records
    ]
    results_df = pd.DataFrame(safe_records)
    results_df.to_csv(f"{output_base}.csv", index=False)
    results_df.to_pickle(f"{output_base}.pkl")
    to_json(safe_records, f"{output_base}.json")
    return results_df


def _run_tool_intent_judge(
    client,
    model_name,
    user_message,
    thinking,
    temperature=0.0,
    max_output_tokens=256,
):
    """LLM-judge call classifying why a tool was invoked, from the user
    message and the model's own reasoning/thoughts.

    KNOWN BUG: references SYSTEM_PROMPT_TOOL_INTENT and
    USER_PROMPT_TOOL_INTENT, neither of which is defined anywhere in
    src/prompts/evaluator_prompts.py or elsewhere in this codebase -- this
    raises NameError the moment it's actually called. Not fixed here: doing
    so would mean writing new LLM-judge prompt content from scratch, which
    isn't a code-cleanup call to make unilaterally. Add those two prompts to
    evaluator_prompts.py before calling this (via
    classify_web_call_tool_intent_from_thoughts()).
    """
    response = client.responses.create(
        model=model_name,
        tools=[],
        tool_choice="none",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_TOOL_INTENT},
            {
                "role": "user",
                "content": USER_PROMPT_TOOL_INTENT.format(
                    user_message=user_message,
                    thinking=thinking,
                ),
            },
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    raw_text = response.output_text
    return {
        "raw_judgment": raw_text,
        "parsed_judgment": _parse_eval_json(raw_text),
    }


def classify_web_call_tool_intent_from_thoughts(
    model_name="gpt-4o-mini",
    input_path=None,
    input_fmt="parquet",
    output_base=None,
    sample_size=None,
    random_state=0,
    only_english=True,
    require_conv_starter=True,
    require_single_user_message=True,
    fallback_to_thoughts_column=False,
    resume=True,
    save_every=25,
    temperature=0.0,
    max_output_tokens=256,
    max_user_message_chars=8000,
    max_thinking_chars=20000,
    make_pie_chart=True,
):
    """
    Find web-call samples with reasoning traces and classify the search intent.

    The judge labels each sample as:
    - Verified Prior Knowledge
    - Acquired New Information
    - Mixed

    When `turn_msgs` is available, the function uses thoughts before the first
    assistant-to-web message/search query. For query-reformulation data, it uses
    only the first `thoughts_list` entry before the first web query. The aggregate
    full-turn `thoughts` column is used only when `fallback_to_thoughts_column`
    is explicitly enabled.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    output_base = _normalise_output_base(output_base, model_name)
    df = _load_tool_intent_input(input_path=input_path, input_fmt=input_fmt)

    original_len = len(df)
    if only_english and "language" in df.columns:
        df = df[df["language"] == "en"].copy()
    if require_conv_starter and "conv_starter" in df.columns:
        df = df[df["conv_starter"] == 1].copy()
    if require_single_user_message and "user_msg_history" in df.columns:
        df = df[
            df["user_msg_history"].apply(
                lambda value: len(_as_list_value(value)) == 1
            )
        ].copy()
    print(f"Loaded {len(df)} samples after filtering out of {original_len}.")

    samples = []
    for source_index, row in df.iterrows():
        if not _row_has_web_call(row):
            continue

        thinking = _thoughts_before_first_web_call(row)
        thinking_source = "turn_msgs_before_first_web_call" if thinking else ""
        if not thinking:
            thinking = _thoughts_before_first_web_query(row)
            thinking_source = "thoughts_list_before_first_web_query" if thinking else ""
        if not thinking and fallback_to_thoughts_column:
            thinking = _stringify_nested_text(row.get("thoughts", ""))
            thinking_source = "thoughts_column_full_turn" if thinking else ""
        if not thinking:
            continue

        user_message = _latest_user_message(row)
        if not user_message:
            continue

        samples.append(
            {
                "source_index": source_index,
                "user_id": row.get("user_id"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "conv_starter": row.get("conv_starter"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "time": row.get("time"),
                "user_message": user_message,
                "thinking": thinking,
                "thinking_source": thinking_source,
            }
        )

    samples_df = pd.DataFrame(samples)
    print(f"Found {len(samples_df)} web-call samples with thoughts.")
    if len(samples_df) == 0:
        return samples_df

    if sample_size is not None and sample_size < len(samples_df):
        samples_df = samples_df.sample(
            n=sample_size,
            random_state=random_state,
        ).reset_index(drop=True)
        print(f"Judging sampled subset of {len(samples_df)} rows.")

    records = []
    completed_keys = set()
    if resume:
        records, completed_keys = _load_existing_tool_intent_records(output_base)
        if records:
            print(f"Loaded {len(records)} existing judgments from {output_base}.csv.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for _, sample in tqdm(samples_df.iterrows(), total=len(samples_df)):
        sample_record = sample.to_dict()
        sample_key = _tool_intent_record_key(sample_record)
        if sample_key in completed_keys:
            continue

        record = {
            **sample_record,
            "judge_model": model_name,
            "temperature": temperature,
        }
        try:
            eval_result = _run_tool_intent_judge(
                client=client,
                model_name=model_name,
                user_message=_trim_text(
                    sample_record["user_message"],
                    max_user_message_chars,
                ),
                thinking=_trim_text(
                    sample_record["thinking"],
                    max_thinking_chars,
                ),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            parsed = eval_result["parsed_judgment"]
            if not isinstance(parsed, dict):
                parsed = {}
            record["tool_intent_label"] = parsed.get("label")
            record["tool_intent_reasoning"] = parsed.get("reasoning")
            record["tool_intent_raw_judgment"] = eval_result["raw_judgment"]
            record["tool_intent_status"] = "ok" if parsed else "parse_failed"
            record["tool_intent_error"] = ""
        except Exception as e:
            print(
                "classify_web_call_tool_intent_from_thoughts",
                sample_record.get("conv_id"),
                sample_record.get("turn_id"),
                e,
            )
            record["tool_intent_label"] = ""
            record["tool_intent_reasoning"] = ""
            record["tool_intent_raw_judgment"] = ""
            record["tool_intent_status"] = "failed"
            record["tool_intent_error"] = str(e)

        records.append(record)
        completed_keys.add(sample_key)
        if save_every and len(records) % save_every == 0:
            _save_tool_intent_records(records, output_base)

    results_df = _save_tool_intent_records(records, output_base)
    if "tool_intent_label" in results_df.columns:
        print("\nTool-intent label counts:")
        print(results_df["tool_intent_label"].value_counts(dropna=False).to_string())
    if make_pie_chart:
        plot_web_call_tool_intent_distribution(
            results_df=results_df,
            output_base=output_base,
            model_name=model_name,
        )
    print(f"Saved judgments to {output_base}.csv/.pkl/.json")
    return results_df


def _load_tool_intent_results(input_path=None, output_base=None, model_name="gpt-4o-mini"):
    if input_path is None:
        output_base = _normalise_output_base(output_base, model_name)
        pkl_path = f"{output_base}.pkl"
        csv_path = f"{output_base}.csv"
        if os.path.exists(pkl_path):
            return pd.read_pickle(pkl_path)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        print(f"Missing tool-intent results at {pkl_path} or {csv_path}.")
        return pd.DataFrame()

    ext = os.path.splitext(str(input_path))[1].lower()
    if ext == ".csv":
        return pd.read_csv(input_path)
    if ext == ".pkl":
        return pd.read_pickle(input_path)
    if ext == ".parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input file type: {ext}")


def plot_web_call_tool_intent_distribution(
    results_df=None,
    input_path=None,
    output_base=None,
    model_name="gpt-4o-mini",
    file_name=None,
):
    if results_df is None:
        results_df = _load_tool_intent_results(
            input_path=input_path,
            output_base=output_base,
            model_name=model_name,
        )
    else:
        results_df = results_df.copy()

    if results_df.empty or "tool_intent_label" not in results_df.columns:
        print("No tool-intent labels found to plot.")
        return pd.DataFrame()

    plot_df = results_df.copy()
    if "tool_intent_status" in plot_df.columns:
        plot_df = plot_df[plot_df["tool_intent_status"] == "ok"].copy()

    label_order = [
        "Verified Prior Knowledge",
        "Acquired New Information",
        "Mixed",
    ]
    labels = plot_df["tool_intent_label"].fillna("").astype(str).str.strip()
    labels = labels[labels != ""]
    if labels.empty:
        print("No non-empty tool-intent labels found to plot.")
        return pd.DataFrame()

    plot_df = (
        labels.value_counts()
        .rename_axis("tool_intent_label")
        .reset_index(name="count")
    )
    plot_df["percentage"] = plot_df["count"] / plot_df["count"].sum()
    plot_df["sort_order"] = plot_df["tool_intent_label"].apply(
        lambda label: (
            label_order.index(label)
            if label in label_order
            else len(label_order)
        )
    )
    plot_df = plot_df.sort_values(
        ["sort_order", "tool_intent_label"]
    ).drop(columns=["sort_order"])

    colors = {
        "Verified Prior Knowledge": "#4C78A8",
        "Acquired New Information": "#F58518",
        "Mixed": "#54A24B",
    }
    marker_colors = [
        colors.get(label, "#BDBDBD")
        for label in plot_df["tool_intent_label"]
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=plot_df["tool_intent_label"],
                values=plot_df["count"],
                customdata=plot_df["percentage"],
                textinfo="label+percent",
                textposition="inside",
                textfont=dict(size=20),
                sort=False,
                hole=0.0,
                showlegend=False,
                marker=dict(colors=marker_colors),
                hovertemplate=(
                    "%{label}<br>"
                    "Count: %{value}<br>"
                    "Share: %{customdata:.1%}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Why the Assistant Called the Web",
        width=900,
        height=700,
        margin=dict(t=80, b=40, l=40, r=40),
        showlegend=False,
    )

    if file_name is None:
        safe_model_name = str(model_name).replace("/", "_")
        file_name = f"web_call_tool_intent_distribution_{safe_model_name}"
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(22, 16))
    try:
        fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
    except Exception as e:
        first_error_line = str(e).strip().splitlines()[0] if str(e).strip() else e
        print(f"Could not write PDF pie chart; HTML was saved. {first_error_line}")

    print("\nTool-intent distribution:")
    print(plot_df.to_string(index=False))
    return plot_df


def subset_selection_for_policy_evaluation_by_human():
    df = pd.read_csv(f"{OUTPUT_PATH}/chatgpt/metadata/web_calls_characterization.csv").reset_index()
    subset = df.sample(100)
    subset.to_csv(f"{OUTPUT_PATH}/chatgpt/metadata/web_calls_characterization_subset_for_human_eval.csv", index=False)


def count_model_used(web_df):
    # Count model usage across all web-call turns. A turn can list multiple models,
    # so we report both total mentions and the final/primary model per turn.
    df = web_df.copy()

    if "openai_models" not in df.columns:
        print("Column `openai_models` not found.")
        return pd.DataFrame(), pd.DataFrame()

    def _ensure_model_list(value):
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str) and v.strip()]
        if pd.isna(value):
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [v for v in parsed if isinstance(v, str) and v.strip()]
            except Exception:
                pass
            return [text]
        return []

    df["openai_models"] = df["openai_models"].apply(_ensure_model_list)
    df["primary_model"] = df["openai_models"].apply(_primary_model)

    all_model_counts = (
        df["openai_models"]
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="count")
    )

    primary_model_counts = (
        df["primary_model"]
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="count")
    )

    print("\nModel mentions across `openai_models`:")
    print(all_model_counts.to_string(index=False))

    print("\nPrimary model per turn:")
    print(primary_model_counts.to_string(index=False))

    return all_model_counts, primary_model_counts


if __name__ == "__main__":
    full_df = load_whole_data_from_file(fmt="pkl")
    print("# all turns:", len(full_df))
    web_df = load_web_data_from_file(fmt="pkl")
    print("# turns with web call:", len(web_df))

    web_call_trend_over_time(full_df)
    web_call_trend_over_time_all_convai(full_df)
    web_call_trend_over_time_by_model(full_df)
    topic_distriction_of_whole_data(full_df)
    topic_distribution_of_web_data(web_df)
    count_model_used(web_df)
    topic_prompt_volume_and_web_rate_over_time(full_df, top_k=5)
