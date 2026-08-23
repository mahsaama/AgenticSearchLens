"""§4 analyses: how conversational prompts become Web queries, and how those
queries evolve across iterations -- fan-out (parallel) vs. sequential
queries, keyword provenance (user prompt / conversation history / prior
search results / parametric knowledge), query-term-count trends over time,
query specificity growth (temporal/geographic/entity), why an agent issues
another query, and user-vs-web-query type/relation classification.

This module is written for the paper's full donated cohort (cross-platform
comparisons, longitudinal trends across hundreds of users, LLM-judge
classification) -- most functions expect the extracted invivo/invitro
dataframes and reference outputs the paper's own pipeline produces (e.g.
replay files under outputs/replays/). It's organized as a large library of
individually-runnable analysis functions rather than one linear script: see
the (mostly commented-out) call list in `if __name__ == "__main__"` at the
bottom for how they're normally invoked one at a time, each writing its own
figure/table under outputs/query_reformulations/.
"""

import os
from dotenv import load_dotenv

load_dotenv()
import ast
import json
from tqdm import tqdm
import pandas as pd
from urllib.parse import urlparse
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

pio.defaults.mathjax = None
from src.utils.common_io import *
from src.utils.chatgpt_conversation_utils import *
from src.utils.figure_style import with_paper_style, styler
from src.web_search_decision.extraction import load_web_data_from_file
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from langdetect import detect
from functools import lru_cache
import numpy as np
from collections import Counter
from src.utils.llm_judge import run_judge
from src.prompts.evaluator_prompts import *


CONF = "./query_reformulations"


SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
}

SNOWBALL_LANGS = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "nl": "dutch",
    "ru": "russian",
    "pt": "portuguese",
}

@lru_cache(maxsize=None)
def get_spacy_model(lang_code):
    if lang_code in SPACY_MODELS:
        return spacy.load(SPACY_MODELS[lang_code], disable=["parser", "ner"])
    return None


@lru_cache(maxsize=None)
def get_stemmer(lang_code):
    if lang_code in SNOWBALL_LANGS:
        return SnowballStemmer(SNOWBALL_LANGS[lang_code])
    return None


def preprocess_text(text, stem=False):
    try:
        lang = detect(text)
    except:
        lang = "en"

    nlp = get_spacy_model(lang)
    if nlp:
        doc = nlp(text)
        lemmas = [
            t.lemma_.lower()
            for t in doc
            if not t.is_punct and not t.is_space and not t.is_stop
        ]
    else:
        tokens = word_tokenize(text)
        lemmas = [t.lower() for t in tokens if t.isalnum()]

    if stem:
        stemmer = get_stemmer(lang)
        if stemmer:
            tokens = [stemmer.stem(t) for t in lemmas]
        else:
            tokens = lemmas
    else:
        tokens = lemmas

    return tokens


def preprocess_text_in_chunks(text, stem=False, max_chunk_chars=100_000):
    text = "" if text is None else str(text)
    if len(text) <= max_chunk_chars:
        return preprocess_text(text, stem=stem)

    tokens = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            tokens.extend(preprocess_text(chunk, stem=stem))

        start = end
        while start < len(text) and text[start].isspace():
            start += 1

    return tokens


def preprocess_texts(texts, stem=False, max_chunk_chars=100_000):
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]

    tokens = []
    for text in texts:
        tokens.extend(
            preprocess_text_in_chunks(
                text,
                stem=stem,
                max_chunk_chars=max_chunk_chars,
            )
        )
    return tokens


def gather_query_reform_effective_factors(df):
    df["thoughts_list"] = [[]] * len(df)
    df["web_queries"] = [[]] * len(df)
    df["sources"] = [[]] * len(df)
    df["memories"] = [[]] * len(df)

    for i, row in tqdm(df.iterrows()):
        all_web_queries = []
        all_thoughts = []
        all_retrieved_sources = []
        all_memories = []

        msgs = json.loads(row["turn_msgs"])
        web_queries = []
        retrieved_sources = []
        thoughts = []
        memories = []

        for msg in msgs:
            # thoughts
            thinking_type = msg["content"].get("content_type", None)
            thinking_thoughts = msg["content"].get("thoughts", [])
            if thinking_type == "thoughts":
                for tt in thinking_thoughts:
                    thoughts.append(tt.get("content", ""))

            # retrieved sources
            retrieved = msg.get("metadata", {}).get("search_result_groups", [])
            for r in retrieved:
                entries = r.get("entries", [])
                for entry in entries:
                    url = entry.get("url", "")
                    domain = urlparse(entry["url"]).netloc.replace("www.", "")
                    title = entry.get("title", "")
                    snippet = entry.get("snippet", "")
                    retrieved_sources.append(f"{title}\n{domain}\n{snippet}")

            retrieved = msg.get("metadata", {}).get("image_results", [])
            for r in retrieved:
                domain = urlparse(entry["url"]).netloc.replace("www.", "")
                title = entry.get("title", "")
                snippet = entry.get("snippet", "")
                retrieved_sources.append(f"{title}\n{domain}\n{snippet}")

            # memories
            memory = (
                msg.get("metadata", {})
                .get("user_context_message_data", {})
                .get("about_user_message", "")
            )
            if memory:
                memories.append(memory)

            # web queries
            search_queries = msg.get("metadata", {}).get("search_queries", [])
            for search_query in search_queries:
                web_queries.append(search_query["q"])
            web_queries += (
                msg.get("metadata", {})
                .get("search_model_queries", {})
                .get("queries", [])
            )
            dedeup_web_queries = list(set(web_queries))
            if dedeup_web_queries:
                all_web_queries.append(dedeup_web_queries)
                all_thoughts.append(thoughts)
                all_retrieved_sources.append(retrieved_sources)
                all_memories.append(memories)
                web_queries = []
                retrieved_sources = []
                thoughts = []
                memories = []

        df.loc[i, "thoughts_list"] = json.dumps(all_thoughts)
        df.loc[i, "web_queries"] = json.dumps(all_web_queries)
        df.loc[i, "sources"] = json.dumps(all_retrieved_sources)
        df.loc[i, "memories"] = json.dumps(all_memories)

    df.drop(columns=["turn_msgs"], inplace=True)
    df.to_csv(
        f"{OUTPUT_PATH}/chatgpt/metadata/query_reformulation_with_thought_src_mem.csv",
        index=False,
    )
    df.reset_index().to_pickle(
        f"{OUTPUT_PATH}/chatgpt/metadata/query_reformulation_with_thought_src_mem.pkl"
    )


# ============================================================
# _other_platforms: Ported from our internal repo (claude/grok/deepseek).
# Public entry is `gather_query_reform_effective_factors_other_platforms(df, platform)`.
# Per-platform helpers are prefixed with `_` to mark as internal; each
# writes to its own outputs/<platform>/metadata/ (previously all three, plus
# the chatgpt-only gather_query_reform_effective_factors() above, wrote to
# the same path and silently overwrote each other/ChatGPT's output).
# ============================================================


def gather_query_reform_effective_factors_other_platforms(df, platform):
    """Dispatcher for non-chatgpt platforms."""
    if platform == "claude":
        return _gather_query_reform_effective_factors_claude(df)
    if platform == "grok":
        return _gather_query_reform_effective_factors_grok(df)
    if platform == "deepseek":
        return _gather_query_reform_effective_factors_deepseek(df)
    raise ValueError(
        f"Unknown platform: {platform!r}. Use 'claude', 'grok', or 'deepseek'."
    )


def _gather_query_reform_effective_factors_claude(df):
    """Claude — web_search only (web_fetch excluded). tool_use ↔ tool_result
    paired via FIFO order within a turn (Claude exports have `id`/`tool_use_id`
    set to None, so ID matching is unavailable)."""
    df["thoughts_list"] = [[]] * len(df)
    df["web_queries"] = [[]] * len(df)
    df["sources"] = [[]] * len(df)
    df["memories"] = [[]] * len(df)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        all_web_queries = []
        all_thoughts = []
        all_retrieved_sources = []
        pending_round_idx = []

        msgs = json.loads(row["turn_msgs"])
        for msg in msgs:
            if msg.get("sender") != "assistant":
                continue

            for b in msg.get("content") or []:
                btype = b.get("type")

                if btype == "tool_use" and b.get("name") == "web_search":
                    q = (b.get("input") or {}).get("query")
                    if not q:
                        continue
                    all_web_queries.append([q])
                    all_thoughts.append([])
                    all_retrieved_sources.append([])
                    pending_round_idx.append(len(all_web_queries) - 1)

                elif btype == "tool_result" and b.get("name") == "web_search":
                    if not pending_round_idx:
                        continue
                    round_idx = pending_round_idx.pop(0)
                    for item in b.get("content") or []:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") not in ("text", "knowledge"):
                            continue
                        url = item.get("url", "") or ""
                        title = item.get("title", "") or ""
                        txt = item.get("text", "") or ""
                        if url or title or txt:
                            all_retrieved_sources[round_idx].append(
                                f"{title}\n{url}\n{txt}"
                            )

        df.loc[i, "thoughts_list"] = json.dumps(all_thoughts)
        df.loc[i, "web_queries"] = json.dumps(all_web_queries)
        df.loc[i, "sources"] = json.dumps(all_retrieved_sources)
        df.loc[i, "memories"] = json.dumps([])

    df.drop(columns=["turn_msgs"], inplace=True)
    df.to_csv(
        f"{OUTPUT_PATH}/claude/metadata/query_reformulation_with_thought_src_mem.csv",
        index=False,
    )
    df.reset_index().to_pickle(
        f"{OUTPUT_PATH}/claude/metadata/query_reformulation_with_thought_src_mem.pkl"
    )


def _gather_query_reform_effective_factors_grok(df):
    """Grok — steps[].tool_usage_cards[].tool.WebSearch. Per-message memory
    references come from metadata.memoryReferences[].summary."""
    df["thoughts_list"] = [[]] * len(df)
    df["web_queries"] = [[]] * len(df)
    df["sources"] = [[]] * len(df)
    df["memories"] = [[]] * len(df)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        all_web_queries = []
        all_thoughts = []
        all_retrieved_sources = []
        all_memories = []

        msgs = json.loads(row["turn_msgs"])
        for msg in msgs:
            if str(msg.get("sender", "")).lower() != "assistant":
                continue

            msg_memories = [
                ref.get("summary", "")
                for ref in ((msg.get("metadata") or {}).get("memoryReferences") or [])
                if ref.get("summary")
            ]

            seen_ids = set()
            for step in msg.get("steps") or []:
                round_queries = []
                round_thoughts = []
                round_sources = []

                for card in step.get("tool_usage_cards") or []:
                    cid = card.get("tool_usage_card_id")
                    tool_obj = card.get("tool") or {}
                    if not (cid and cid not in seen_ids and "WebSearch" in tool_obj):
                        continue
                    seen_ids.add(cid)
                    q = tool_obj["WebSearch"].get("args", {}).get("query")
                    if q:
                        round_queries.append(q)

                for result_wrap in step.get("tool_usage_results") or []:
                    result = result_wrap.get("result") or {}
                    for r in result.get("WebSearchResults") or []:
                        title = r.get("title") or ""
                        url = r.get("url") or ""
                        preview = r.get("preview") or ""
                        if title or url or preview:
                            round_sources.append(f"{title}\n{url}\n{preview}")

                if round_queries:
                    all_web_queries.append(round_queries)
                    all_thoughts.append(round_thoughts)
                    all_retrieved_sources.append(round_sources)
                    all_memories.append(msg_memories)

        df.loc[i, "thoughts_list"] = json.dumps(all_thoughts)
        df.loc[i, "web_queries"] = json.dumps(all_web_queries)
        df.loc[i, "sources"] = json.dumps(all_retrieved_sources)
        df.loc[i, "memories"] = json.dumps(all_memories)

    df.drop(columns=["turn_msgs"], inplace=True)
    df.to_csv(
        f"{OUTPUT_PATH}/grok/metadata/query_reformulation_with_thought_src_mem.csv",
        index=False,
    )
    df.reset_index().to_pickle(
        f"{OUTPUT_PATH}/grok/metadata/query_reformulation_with_thought_src_mem.pkl"
    )

def _gather_query_reform_effective_factors_deepseek(df):
    """DeepSeek — SEARCH fragments expose only integer query_indexes.
    Placeholder strings 'Query {idx} of Iter {n}' are synthesized to keep the
    web_queries schema uniform. Do NOT feed placeholders to similarity/LLM-judge."""
    df["thoughts_list"] = [[]] * len(df)
    df["web_queries"] = [[]] * len(df)
    df["sources"] = [[]] * len(df)
    df["memories"] = [[]] * len(df)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        all_web_queries = []
        all_thoughts = []
        all_retrieved_sources = []

        msgs = json.loads(row["turn_msgs"])
        for node in msgs:
            msg = node.get("message") or {}
            msg_queries = []
            msg_sources = []

            for frag in msg.get("fragments") or []:
                if frag.get("type") != "SEARCH":
                    continue
                round_query_idxs = set()
                round_sources = []
                for r in frag.get("results") or []:
                    for qi in r.get("query_indexes") or []:
                        round_query_idxs.add(qi)
                    url = r.get("url") or ""
                    title = r.get("title") or ""
                    snippet = r.get("snippet") or ""
                    if title or url or snippet:
                        round_sources.append(f"{title}\n{url}\n{snippet}")
                if round_query_idxs:
                    iter_num = len(msg_queries) + 1
                    round_query_strings = [
                        f"Query {qi} of Iter {iter_num}"
                        for qi in sorted(round_query_idxs)
                    ]
                    msg_queries.append(round_query_strings)
                    msg_sources.append(round_sources)

            if len(msg_queries) > len(all_web_queries):
                all_web_queries = msg_queries
                all_retrieved_sources = msg_sources
                all_thoughts = [[] for _ in msg_queries]

        df.loc[i, "thoughts_list"] = json.dumps(all_thoughts)
        df.loc[i, "web_queries"] = json.dumps(all_web_queries)
        df.loc[i, "sources"] = json.dumps(all_retrieved_sources)
        df.loc[i, "memories"] = json.dumps([])

    df.drop(columns=["turn_msgs"], inplace=True)
    df.to_csv(
        f"{OUTPUT_PATH}/deepseek/metadata/query_reformulation_with_thought_src_mem.csv",
        index=False,
    )
    df.reset_index().to_pickle(
        f"{OUTPUT_PATH}/deepseek/metadata/query_reformulation_with_thought_src_mem.pkl"
    )

def web_query_tokens_source_detection(platform="chatgpt"):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    df = _add_web_query_token_source_columns(df)

    df.reset_index(drop=True, inplace=True)
    _save_dataframe(df, "query_reformulation_with_thought_src_mem_v3", platform=platform)


DEFAULT_WEB_QUERY_TOKEN_SOURCE_FACTOR_COLS = {
    "all_new_words_from_user_queries": "Latest<br>User Prompt",
    "all_new_words_from_assistant_queries": "Conversation<br>History",
    "all_new_words_from_memories": "Other<br>Conversations",
    "all_new_words_from_sources": "Search<br>Results",
    "all_new_words_from_pk": "Parametric<br>Knowledge<br>[Potentially]",
}


def _plot_web_query_tokens_source_detection_from_df(
    df,
    factor_cols=None,
    base_file_name="web_query_token_source_detection",
    skip_source_for_one_loop=True,
    platform="chatgpt",
):
    factor_cols = dict(factor_cols or DEFAULT_WEB_QUERY_TOKEN_SOURCE_FACTOR_COLS)
    user_prompt_col = "all_new_words_from_user_queries"
    history_col = "all_new_words_from_assistant_queries"
    memories_col = "all_new_words_from_memories"
    unexplained_col = "all_new_words_from_pk"

    df = df.copy()
    for col in {"web_queries", "memories"} | set(factor_cols.keys()):
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].apply(lambda x: _safe_json_value(x, []))

    def _flatten_nested_texts(values):
        if not isinstance(values, list):
            values = [values]
        flat_values = []
        for item in values:
            if isinstance(item, list):
                flat_values.extend(item)
            else:
                flat_values.append(item)
        return [item.strip() for item in flat_values if isinstance(item, str) and item.strip()]

    num_rows_with_memories = int(
        df["memories"].apply(lambda items: len(_flatten_nested_texts(items)) > 0).sum()
    )
    num_memory_keyword_hits = int(
        df["all_new_words_from_memories"].apply(lambda items: len(items) > 0).sum()
    )

    print(f"Rows with memories: {num_rows_with_memories}")
    print(f"Memory keywords in web queries: {num_memory_keyword_hits}")

    plot_factor_cols = {
        col: label for col, label in factor_cols.items() if col != memories_col
    }

    plot_df = df.copy()
    plot_df[history_col] = plot_df.apply(
        lambda row: list(set(row[history_col]) | set(row[user_prompt_col])),
        axis=1,
    )
    plot_df[unexplained_col] = plot_df.apply(
        lambda row: list(set(row[unexplained_col]) | set(row[memories_col])),
        axis=1,
    )

    plot_df["num_web_query_words"] = plot_df.apply(
        lambda row: len(set().union(*(set(row[col]) for col in plot_factor_cols))),
        axis=1,
    )
    plot_df["num_loops"] = plot_df["web_queries"].apply(len)
    plot_df = plot_df[plot_df["num_web_query_words"] > 0].copy()

    for col in plot_factor_cols:
        rate_col = f"{col}_rate"
        plot_df[rate_col] = plot_df.apply(
            lambda row: len(row[col]) / row["num_web_query_words"]
            if row["num_web_query_words"] > 0
            else np.nan,
            axis=1,
        )

    def _plot_subset(subset_df, file_name, title_suffix):
        if len(subset_df) == 0:
            return

        fig = go.Figure()
        for i, (col, label) in enumerate(plot_factor_cols.items()):
            if (
                skip_source_for_one_loop
                and "1_loop" in file_name
                and col == "all_new_words_from_sources"
            ):
                continue
            fig.add_trace(
                go.Box(
                    y=subset_df[f"{col}_rate"],
                    name=label,
                    marker_color=px.colors.qualitative.Plotly[i],
                    line_color=px.colors.qualitative.Plotly[i],
                    showlegend=False,
                    boxmean=True,
                )
            )

        fig.update_layout(
            xaxis_title="Keyword Source",
            yaxis_title="Share of Web Query Keywords",
            title=title_suffix,
            xaxis=dict(
                tickangle=0,
            )
        )
        fig.update_yaxes(tickformat=".0%")

        os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
        fig = with_paper_style(fig, config=styler(22, 14))
        fig.update_xaxes(tickfont=dict(size=22))
        fig.update_yaxes(tickfont=dict(size=22))
        fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf")

    def _plot_bar_subset(subset_df, file_name, title_suffix):
        if len(subset_df) == 0:
            return

        subset_factor_cols = {
            col: label
            for col, label in plot_factor_cols.items()
            if not (
                skip_source_for_one_loop
                and "1_loop" in file_name
                and col == "all_new_words_from_sources"
            )
        }
        factor_counts = {
            col: subset_df[col].apply(lambda words: len(set(words))).sum()
            for col in subset_factor_cols
        }
        total_count = sum(factor_counts.values())
        if total_count == 0:
            return

        labels = list(subset_factor_cols.values())
        shares = [factor_counts[col] / total_count for col in subset_factor_cols]
        colors = px.colors.qualitative.Plotly[: len(subset_factor_cols)]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=shares,
                text=[f"{share:.1%}" for share in shares],
                textposition="outside",
                marker_color=colors,
                showlegend=False,
            )
        )
        fig.update_layout(
            xaxis_title="Keyword Source",
            yaxis_title="Share of Web Query Keywords",
            title=title_suffix,
        )
        fig.update_yaxes(tickformat=".0%", range=[0, max(shares) * 1.15])

        os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
        fig = with_paper_style(fig, config=styler(18, 14))
        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=18))
        fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf")

    one_loop_df = plot_df[plot_df["num_loops"] == 1].copy()
    multi_loop_df = plot_df[plot_df["num_loops"] > 1].copy()

    _plot_subset(
        one_loop_df,
        f"{base_file_name}_1_loop",
        "1 Web Query Loop",
    )
    _plot_subset(
        multi_loop_df,
        f"{base_file_name}_multi_loop",
        "2+ Web Query Loops",
    )
    _plot_bar_subset(
        one_loop_df,
        f"{base_file_name}_bar_1_loop",
        "1 Web Query Loop",
    )
    _plot_bar_subset(
        multi_loop_df,
        f"{base_file_name}_bar_multi_loop",
        "2+ Web Query Loops",
    )


def plot_web_query_tokens_source_detection(platform="chatgpt"):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem_v3.pkl"
    )
    _plot_web_query_tokens_source_detection_from_df(df, platform=platform)


def _plot_number_of_loops_histogram_from_df(
    df,
    number_file_name="number_of_query_reformulations",
    parallel_file_name="parallel_queries_by_query_reformulations",
    samples_file_name="number_of_query_reformulations_samples",
    sample_language="en",
    drop_zero_loop_rows=True,
    trend_file_name="number_of_query_reformulations_and_parallel_queries_over_time",
    trend_tick_interval_months=2,
    platform="chatgpt",
):
    bucket_order = ([] if drop_zero_loop_rows else ["0"]) + ["1", "2", "3+"]
    parallel_plot_x_order = ["1", "2", "3+"]

    def _bucket_count(value):
        return "3+" if value >= 3 else str(value)

    def _parallel_plot_x_bucket(value):
        return "3+" if value >= 3 else str(value)

    count_n_hops = {}
    loop_samples = {}
    parallel_query_counts = {
        n_hops_bucket: {parallel_bucket: 0 for parallel_bucket in bucket_order}
        for n_hops_bucket in parallel_plot_x_order
    }
    trend_rows = []

    for _, row in tqdm(df.iterrows()):
        all_system_queries = _safe_json_value(row.get("web_queries"), [])
        all_thoughts = _safe_json_value(row.get("thoughts_list"), [])
        all_retrieved_sources = _safe_json_value(row.get("sources"), [])

        n_hops = len(all_system_queries)
        if drop_zero_loop_rows and n_hops == 0:
            continue

        count_n_hops[n_hops] = count_n_hops.get(n_hops, 0) + 1
        bucket = _bucket_count(n_hops)
        max_parallel_queries = max(
            (
                len(query_group) if isinstance(query_group, list) else 1
                for query_group in all_system_queries
            ),
            default=0,
        )
        num_parallel_queries = sum(
            (
                len(query_group) if isinstance(query_group, list) else 1
                for query_group in all_system_queries
            )
        )
        trend_rows.append(
            {
                "time": row.get("time"),
                "num_loops": n_hops,
                "num_parallel_queries": num_parallel_queries,
            }
        )
        if n_hops >= 1 and max_parallel_queries >= 1:
            parallel_query_counts[_parallel_plot_x_bucket(n_hops)][
                _bucket_count(max_parallel_queries)
            ] += 1
        sample_language_match = (
            sample_language is None or row.get("language") == sample_language
        )
        if n_hops >= 1 and bucket not in loop_samples and sample_language_match:
            user_msg_history = _safe_json_value(row.get("user_msg_history"), [])
            if isinstance(user_msg_history, str):
                user_msg_history = [user_msg_history]
            loop_samples[bucket] = {
                "num_loops_bucket": bucket,
                "num_loops": n_hops,
                "user_query": str(user_msg_history[-1]).strip() if user_msg_history else "",
                "web_queries": all_system_queries,
                "thoughts": all_thoughts,
                "srcs_retrieved": all_retrieved_sources,
                "result_key": row.get("result_key"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                # "topic": row.get("topic"),
            }

    print(count_n_hops)

    count_n_hops_sum = sum(count_n_hops.values())
    print(count_n_hops_sum)
    if count_n_hops_sum == 0:
        print("No web query loops to plot.")
        return {
            "count_n_hops": count_n_hops,
            "parallel_query_counts": parallel_query_counts,
            "loop_samples": loop_samples,
            "time_trend": [],
        }

    binned_percentages = {bucket: 0.0 for bucket in bucket_order}
    for n_hops, count in count_n_hops.items():
        percentage = 100 * count / count_n_hops_sum
        if n_hops >= 3:
            binned_percentages["3+"] += percentage
        elif n_hops >= 1:
            binned_percentages[str(n_hops)] += percentage
        elif "0" in binned_percentages:
            binned_percentages["0"] += percentage

    x = list(binned_percentages.keys())
    y = [round(binned_percentages[label], 2) for label in x]
    text = [f"{value:.1f}%" for value in y]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            text=text,
            textposition="outside",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Number of Query Formulations",
        yaxis_title="Turns (%)",
        yaxis=dict(range=[0, max(y) * 1.15 if y else 1]),
    )
    fig.update_yaxes(ticksuffix="%")
    os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
    file_name = number_file_name
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))
    fig.write_image(
        f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf"
    )
    to_json(
        [
            loop_samples[label]
            for label in x
            if label in loop_samples
        ],
        f"{OUTPUT_PATH}/{platform}/{CONF}/{samples_file_name}.json",
    )

    fig = go.Figure()
    for parallel_bucket in bucket_order:
        y = [
            round(
                100
                * parallel_query_counts[n_hops_bucket][parallel_bucket]
                / count_n_hops_sum,
                2,
            )
            for n_hops_bucket in parallel_plot_x_order
        ]
        fig.add_trace(
            go.Bar(
                x=parallel_plot_x_order,
                y=y,
                name=parallel_bucket,
                text=[f"{value:.1f}%" if value > 0 else "" for value in y],
                textposition="outside",
                hovertemplate=(
                    "Query formulations: %{x}<br>"
                    f"Max Fan-out queries: {parallel_bucket}<br>"
                    "Turns: %{customdata}<br>"
                    "Share: %{y:.2f}%<extra></extra>"
                ),
                customdata=[
                    parallel_query_counts[n_hops_bucket][parallel_bucket]
                    for n_hops_bucket in parallel_plot_x_order
                ],
            )
        )
    max_breakdown_y = max(
        [
            100 * count / count_n_hops_sum
            for n_hops_bucket in parallel_plot_x_order
            for count in parallel_query_counts[n_hops_bucket].values()
        ]
        or [0]
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Number of Query Formulations",
        yaxis_title="Turns (%)",
        yaxis=dict(range=[0, max_breakdown_y * 1.25 if max_breakdown_y else 1]),
        legend_title_text="Max Fan-out Queries",
    )
    fig.update_yaxes(ticksuffix="%")
    file_name = parallel_file_name
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18))
    fig.write_image(
        f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf"
    )

    trend_records = []
    if trend_file_name:
        trend_df = pd.DataFrame(trend_rows)
        if "time" in trend_df.columns:
            trend_df["time"] = pd.to_datetime(trend_df["time"], errors="coerce")
            trend_df = trend_df.dropna(subset=["time"])

        if len(trend_df) > 0:
            trend_df["month"] = trend_df["time"].dt.to_period("M").dt.to_timestamp()
            monthly_trends = (
                trend_df.groupby("month")
                .agg(
                    avg_num_loops=("num_loops", "mean"),
                    avg_num_parallel_queries=("num_parallel_queries", "mean"),
                    num_turns=("num_loops", "size"),
                )
                .reset_index()
                .sort_values("month")
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends["month"],
                    y=monthly_trends["avg_num_loops"],
                    mode="lines+markers",
                    name="Number of Iterations",
                    customdata=monthly_trends["num_turns"],
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        "Average: %{y:.2f}<br>"
                        "Turns: %{customdata}<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=monthly_trends["month"],
                    y=monthly_trends["avg_num_parallel_queries"],
                    mode="lines+markers",
                    name="Number of Fan-out Queries",
                    customdata=monthly_trends["num_turns"],
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        "Average: %{y:.2f}<br>"
                        "Turns: %{customdata}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Average Number",
                xaxis=dict(
                    tickmode="linear",
                    dtick=f"M{trend_tick_interval_months}",
                    tickformat="%b %Y",
                    tickangle=-45,
                ),
                margin=dict(b=90),
            )
            file_name = trend_file_name
            fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.9, 1.2))
            fig.write_image(
                f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf"
            )

            trend_records = monthly_trends.copy()
            trend_records["month"] = trend_records["month"].dt.strftime("%Y-%m")
            trend_records = trend_records.to_dict(orient="records")
        else:
            print("No valid timestamps found for loop trend plot.")

    return {
        "count_n_hops": count_n_hops,
        "parallel_query_counts": parallel_query_counts,
        "loop_samples": loop_samples,
        "time_trend": trend_records,
    }


def plot_number_of_loops_histogram(platform="chatgpt"):
    platform_configs = [
        ("openai", "ChatGPT"),
        ("claude", "Claude"),
        ("grok", "Grok"),
        ("deepseek", "DeepSeek"),
    ]
    bucket_order = ["1", "2", "3+"]
    bucket_colors = {
        "1": "#636EFA",
        "2": "#EF553B",
        "3+": "#00CC96",
    }

    def _platform_candidate_paths(platform):
        # platform_configs' "openai" entry maps to outputs/chatgpt/ -- this
        # file's folder convention is "chatgpt", not "openai".
        platform_dir = "chatgpt" if platform == "openai" else platform
        if platform == "openai":
            return [
                f"{OUTPUT_PATH}/{platform_dir}/metadata/query_reformulation_with_thought_src_mem_v2.pkl",
            ]
        return [
            f"{OUTPUT_PATH}/{platform_dir}/metadata/query_reformulation_with_thought_src_mem.pkl",
        ]

    def _load_platform_df(platform):
        for candidate_path in _platform_candidate_paths(platform):
            if not os.path.exists(candidate_path):
                continue
            try:
                return pd.read_pickle(candidate_path).copy(), candidate_path
            except Exception as e:
                print(f"Failed to load `{candidate_path}`: {e}")
        print(f"No query reformulation metadata file found for `{platform}`.")
        return None, None

    def _bucket_count(value):
        return "3+" if value >= 3 else str(value)

    def _compute_parallel_query_payload(df):
        count_n_hops = {}
        parallel_query_counts = {
            n_hops_bucket: {parallel_bucket: 0 for parallel_bucket in bucket_order}
            for n_hops_bucket in bucket_order
        }

        for _, row in tqdm(df.iterrows(), total=len(df)):
            web_query_groups = _safe_json_value(row.get("web_queries"), [])
            if not isinstance(web_query_groups, list):
                continue
            n_hops = len(web_query_groups)
            if n_hops == 0:
                continue

            count_n_hops[n_hops] = count_n_hops.get(n_hops, 0) + 1
            max_parallel_queries = max(
                (
                    len(query_group) if isinstance(query_group, list) else 1
                    for query_group in web_query_groups
                ),
                default=0,
            )
            if max_parallel_queries >= 1:
                parallel_query_counts[_bucket_count(n_hops)][
                    _bucket_count(max_parallel_queries)
                ] += 1

        count_n_hops_sum = sum(count_n_hops.values())
        if count_n_hops_sum == 0:
            return None

        max_breakdown_y = max(
            [
                100 * count / count_n_hops_sum
                for n_hops_bucket in bucket_order
                for count in parallel_query_counts[n_hops_bucket].values()
            ]
            or [0]
        )
        return {
            "count_n_hops_sum": count_n_hops_sum,
            "parallel_query_counts": parallel_query_counts,
            "max_breakdown_y": max_breakdown_y,
        }

    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    )
    results = _plot_number_of_loops_histogram_from_df(df, platform=platform)

    subplot_payloads = []
    platform_data_sources = {}
    for platform, display_name in platform_configs:
        platform_df, source_path = _load_platform_df(platform)
        if platform_df is None:
            continue
        payload = _compute_parallel_query_payload(platform_df)
        if payload is None:
            print(f"No web query loops to plot for `{platform}`.")
            continue
        payload["platform"] = platform
        payload["display_name"] = display_name
        subplot_payloads.append(payload)
        platform_data_sources[platform] = source_path

    if subplot_payloads:
        combined_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[item["display_name"] for item in subplot_payloads],
            vertical_spacing=0.2,
            horizontal_spacing=0.08,
        )
        combined_fig.update_annotations(font_size=22)
        legend_shown = set()

        for panel_idx, panel in enumerate(subplot_payloads):
            row_idx = panel_idx // 2 + 1
            col_idx = panel_idx % 2 + 1
            for parallel_bucket in bucket_order:
                y = [
                    round(
                        100
                        * panel["parallel_query_counts"][n_hops_bucket][parallel_bucket]
                        / panel["count_n_hops_sum"],
                        2,
                    )
                    for n_hops_bucket in bucket_order
                ]
                showlegend = parallel_bucket not in legend_shown
                if showlegend:
                    legend_shown.add(parallel_bucket)
                combined_fig.add_trace(
                    go.Bar(
                        x=bucket_order,
                        y=y,
                        name=parallel_bucket,
                        marker_color=bucket_colors[parallel_bucket],
                        text=[f"{value:.1f}%" if value > 0 else "" for value in y],
                        textposition="outside",
                        textfont=dict(size=20),
                        customdata=[
                            panel["parallel_query_counts"][n_hops_bucket][parallel_bucket]
                            for n_hops_bucket in bucket_order
                        ],
                        hovertemplate=(
                            "Query formulations: %{x}<br>"
                            f"Max Fan-out queries: {parallel_bucket}<br>"
                            "Turns: %{customdata}<br>"
                            "Share: %{y:.2f}%<extra></extra>"
                        ),
                        showlegend=showlegend,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            combined_fig.update_yaxes(
                range=[
                    0,
                    panel["max_breakdown_y"] * 1.25
                    if panel["max_breakdown_y"]
                    else 1,
                ],
                ticksuffix="%",
                row=row_idx,
                col=col_idx,
            )

        combined_fig.update_layout(
            barmode="group",
            legend_title_text="Max Fan-out Queries",
            title="Parallel Queries by Query Reformulations",
            margin=dict(t=110, b=90, l=95, r=35),
        )
        combined_fig.add_annotation(
            text="Number of Query Formulations",
            x=0.5,
            y=-0.25,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=24),
        )
        combined_fig.add_annotation(
            text="Turns (%)",
            x=-0.16,
            y=0.5,
            xref="paper",
            yref="paper",
            textangle=-90,
            showarrow=False,
            font=dict(size=24),
        )
        # Combined all-platform comparison -- stays flat, not nested under
        # the single `platform` this function's histogram half uses.
        os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)
        combined_fig = with_paper_style(
            combined_fig,
            config=styler(20, 20),
            legend_pos=(0.8, 1.3),
        )
        combined_fig.write_image(
            f"{OUTPUT_PATH}/{CONF}/parallel_queries_by_query_reformulations_all_models.pdf",
            format="pdf",
        )
        to_json(
            {
                "platforms_plotted": [item["platform"] for item in subplot_payloads],
                "platform_data_sources": platform_data_sources,
            },
            f"{OUTPUT_PATH}/metadata/parallel_queries_by_query_reformulations_all_models_sources.json",
        )

    return results


def plot_number_of_query_reformulations_over_time(
    drop_zero_loop_rows=True,
    file_name="number_of_query_reformulations_over_time",
    tick_interval_months=2,
    platform="chatgpt",
):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    ).copy()

    if "web_queries" not in df.columns:
        print("Column `web_queries` not found.")
        return []

    df["web_queries"] = df["web_queries"].apply(lambda x: _safe_json_value(x, []))
    df["num_loops"] = df["web_queries"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if drop_zero_loop_rows:
        df = df[df["num_loops"] > 0].copy()

    df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    if len(df) == 0:
        print("No valid rows for query reformulation trend plot.")
        return []

    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month")
        .agg(
            avg_num_loops=("num_loops", "mean"),
            num_turns=("num_loops", "size"),
        )
        .reset_index()
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_num_loops"],
            mode="lines+markers",
            name="Number of Query Formulations",
            customdata=monthly["num_turns"],
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Average: %{y:.2f}<br>"
                "Turns: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Number of Query Formulations",
        xaxis=dict(
            tickmode="linear",
            dtick=f"M{tick_interval_months}",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90),
    )
    os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf")

    records = monthly.copy()
    records["month"] = records["month"].dt.strftime("%Y-%m")
    return records.to_dict(orient="records")


def plot_number_of_parallel_queries_over_time(
    drop_zero_loop_rows=True,
    file_name="number_of_parallel_queries_over_time",
    tick_interval_months=2,
    platform="chatgpt",
):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    ).copy()

    if "web_queries" not in df.columns:
        print("Column `web_queries` not found.")
        return []

    df["web_queries"] = df["web_queries"].apply(lambda x: _safe_json_value(x, []))
    df["num_loops"] = df["web_queries"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if drop_zero_loop_rows:
        df = df[df["num_loops"] > 0].copy()

    df["num_parallel_queries"] = df["web_queries"].apply(
        lambda queries: max(
            (
                len(query_group) if isinstance(query_group, list) else 1
                for query_group in queries
            ),
            default=0,
        )
        if isinstance(queries, list)
        else 0
    )
    df["num_total_queries"] = df["web_queries"].apply(
        lambda queries: sum(
            len(query_group) if isinstance(query_group, list) else 1
            for query_group in queries
        )
        if isinstance(queries, list)
        else 0
    )

    df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    if len(df) == 0:
        print("No valid rows for parallel query trend plot.")
        return []

    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month")
        .agg(
            avg_num_parallel_queries=("num_parallel_queries", "mean"),
            num_turns=("num_parallel_queries", "size"),
        )
        .reset_index()
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_num_parallel_queries"],
            mode="lines+markers",
            name="Number of Parallel Queries",
            customdata=monthly["num_turns"],
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Average: %{y:.2f}<br>"
                "Turns: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Number of Parallel Queries",
        xaxis=dict(
            tickmode="linear",
            dtick=f"M{tick_interval_months}",
            tickformat="%b %Y",
            tickangle=-45,
        ),
        margin=dict(b=90),
    )
    os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf")

    records = monthly.copy()
    records["month"] = records["month"].dt.strftime("%Y-%m")
    return records.to_dict(orient="records")


def plot_number_of_fanout_queries_and_iterations_over_time(
    drop_zero_loop_rows=True,
    file_name="number_of_query_reformulations_and_parallel_queries_over_time",
    tick_interval_months=1,
    platform="chatgpt",
):
    xaxis_start = pd.Timestamp("2024-09-01")
    xaxis_end = pd.Timestamp("2026-01-31")
    full_month_range = pd.date_range(start=xaxis_start, end=xaxis_end, freq="MS")
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem_v2.pkl"
    ).copy()

    if "web_queries" not in df.columns:
        print("Column `web_queries` not found.")
        return []

    df["web_queries"] = df["web_queries"].apply(lambda x: _safe_json_value(x, []))
    df["num_loops"] = df["web_queries"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    if drop_zero_loop_rows:
        df = df[df["num_loops"] > 0].copy()

    df["num_parallel_queries"] = df["web_queries"].apply(
        lambda queries: max(
            (
                len(query_group) if isinstance(query_group, list) else 1
                for query_group in queries
            ),
            default=0,
        )
        if isinstance(queries, list)
        else 0
    )
    df["num_total_queries"] = df["web_queries"].apply(
        lambda queries: sum(
            len(query_group) if isinstance(query_group, list) else 1
            for query_group in queries
        )
        if isinstance(queries, list)
        else 0
    )

    df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    if len(df) == 0:
        print("No valid rows for fan-out and iteration trend plot.")
        return []

    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month")
        .agg(
            avg_num_loops=("num_loops", "mean"),
            avg_num_parallel_queries=("num_parallel_queries", "mean"),
            avg_num_total_queries=("num_total_queries", "mean"),
            num_turns=("num_loops", "size"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly = (
        monthly.set_index("month")
        .reindex(full_month_range)
        .fillna(0)
        .rename_axis("month")
        .reset_index()
    )
    monthly["num_turns"] = monthly["num_turns"].astype(int)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_num_loops"],
            mode="lines+markers",
            name="Number of Iterations",
            customdata=monthly["num_turns"],
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Average: %{y:.2f}<br>"
                "Turns: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_num_parallel_queries"],
            mode="lines+markers",
            name="Number of Fan-out Queries",
            customdata=monthly["num_turns"],
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Average: %{y:.2f}<br>"
                "Turns: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_num_total_queries"],
            mode="lines+markers",
            name="Overall Number of Queries",
            customdata=monthly["num_turns"],
            hovertemplate=(
                "Month: %{x|%b %Y}<br>"
                "Average: %{y:.2f}<br>"
                "Turns: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Number",
        xaxis=dict(
            tickmode="linear",
            dtick=f"M{tick_interval_months}",
            tickformat="%b %Y",
            tickangle=-45,
            range=[xaxis_start, xaxis_end],
        ),
        margin=dict(b=90),
    )
    os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
    fig = with_paper_style(fig, config=styler(20, 18), legend_pos=(0.8, 1.3))
    fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{file_name}.pdf", format="pdf")

    records = monthly.copy()
    records["month"] = records["month"].dt.strftime("%Y-%m")
    return records.to_dict(orient="records")


@lru_cache(maxsize=200_000)
def _count_terms(text, remove_stopwords=False):
    text = "" if text is None else str(text).strip()
    if not text:
        return 0
    if remove_stopwords:
        return len(preprocess_text_in_chunks(text))
    return len(text.split())


def _clean_web_query_groups(value):
    parsed = _safe_json_value(value, [])
    if not isinstance(parsed, list):
        return []

    cleaned_groups = []
    for query_group in parsed:
        if isinstance(query_group, list):
            cleaned_group = [
                str(query).strip()
                for query in query_group
                if isinstance(query, str) and str(query).strip()
            ]
        elif isinstance(query_group, str) and query_group.strip():
            cleaned_group = [query_group.strip()]
        else:
            cleaned_group = []

        if cleaned_group:
            cleaned_groups.append(cleaned_group)

    return cleaned_groups


def _count_url_source_records(items):
    if not isinstance(items, list):
        return 0
    return sum(
        1
        for item in items
        if isinstance(item, dict) and item.get("url", "")
    )


def plot_query_term_count_trends_over_time(remove_stopwords=False):
    return _plot_query_term_count_trends_over_time_multiplatform(
        remove_stopwords=remove_stopwords
    )


def _plot_query_term_count_trends_over_time_multiplatform(remove_stopwords=False):
    platform_configs = [
        ("openai", "ChatGPT"),
        ("claude", "Claude"),
        ("grok", "Grok"),
        ("deepseek", "DeepSeek"),
    ]
    iteration_bucket_order = ["1", "2", "3+"]

    def _timeline_iteration_bucket(num_iterations):
        if num_iterations <= 1:
            return "1"
        if num_iterations == 2:
            return "2"
        return "3+"

    def _platform_candidate_paths(platform):
        if platform == "openai":
            return [
                f"{OUTPUT_PATH}/chatgpt/metadata/query_reformulation_with_thought_src_mem_v2.pkl",
            ]

        candidates = [
            f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem.pkl",
        ]
        return candidates

    def _load_platform_df(platform):
        for candidate_path in _platform_candidate_paths(platform):
            if not os.path.exists(candidate_path):
                continue
            try:
                return pd.read_pickle(candidate_path).copy(), candidate_path
            except Exception as e:
                print(f"Failed to load `{candidate_path}`: {e}")

        print(f"No query reformulation metadata file found for `{platform}`.")
        return None, None

    def _platform_response_source_paths(platform):
        if platform == "openai":
            return [
                f"{OUTPUT_PATH}/chatgpt/metadata/response_and_sources.pkl",
                f"{OUTPUT_PATH}/chatgpt/metadata/response_and_sources.csv",
            ]
        return [
            f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources.pkl",
            f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources.csv",
        ]

    def _load_retrieved_url_count_lookup(platform):
        source_df, _source_path = _load_dataframe_from_candidates(
            _platform_response_source_paths(platform)
        )
        if source_df is None:
            return {}

        required_cols = {"user_id", "conv_id", "turn_id", "srcs_retrieved"}
        missing_cols = required_cols - set(source_df.columns)
        if missing_cols:
            print(
                f"Missing response/source columns for `{platform}`: {sorted(missing_cols)}"
            )
            return {}

        lookup = {}
        for _, row in source_df.iterrows():
            key = (
                str(row.get("user_id", "")),
                str(row.get("conv_id", "")),
                str(row.get("turn_id", "")),
            )
            retrieved_items = _safe_json_value(row.get("srcs_retrieved"), [])
            lookup[key] = _count_url_source_records(retrieved_items)
        return lookup

    def _has_query_text_for_term_metrics(platform):
        return platform != "deepseek"

    def _collect_platform_metrics(df, platform, retrieved_url_count_lookup):
        df = df.copy()
        df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
        has_query_text = _has_query_text_for_term_metrics(platform)

        metrics = {
            "terms_per_query": [],
            "terms_per_prompt": [],
            "avg_query_terms_per_prompt": [],
            "total_query_terms_per_prompt": [],
            "total_queries_per_prompt": [],
            "retrieved_urls_per_prompt": [],
            "retrieved_urls_per_web_query": [],
            "iterations_per_prompt": [],
            "iteration_timeline_rows": [],
        }

        for _, row in tqdm(df.iterrows(), total=len(df)):
            web_query_groups = _clean_web_query_groups(row.get("web_queries"))
            if not web_query_groups:
                continue

            flat_web_queries = [q for group in web_query_groups for q in group]
            if len(flat_web_queries) == 0:
                continue

            metrics["total_queries_per_prompt"].append(len(flat_web_queries))
            num_iterations = int(len(web_query_groups))
            metrics["iterations_per_prompt"].append(num_iterations)

            retrieved_item_count = retrieved_url_count_lookup.get(
                (
                    str(row.get("user_id", "")),
                    str(row.get("conv_id", "")),
                    str(row.get("turn_id", "")),
                ),
                0,
            )
            metrics["retrieved_urls_per_prompt"].append(float(retrieved_item_count))
            if retrieved_item_count > 0:
                per_query_retrieved_count = retrieved_item_count / len(flat_web_queries)
                metrics["retrieved_urls_per_web_query"] += [
                    float(per_query_retrieved_count)
                ] * len(flat_web_queries)

            if has_query_text:
                web_query_term_counts = [
                    _count_terms(query, remove_stopwords) for query in flat_web_queries
                ]
                if len(web_query_term_counts) > 0:
                    metrics["terms_per_query"] += web_query_term_counts
                    latest_user_query = _row_latest_user_query(row)
                    metrics["terms_per_prompt"].append(
                        _count_terms(latest_user_query, remove_stopwords)
                    )
                    metrics["avg_query_terms_per_prompt"].append(
                        float(np.mean(web_query_term_counts))
                    )
                    metrics["total_query_terms_per_prompt"].append(
                        float(np.sum(web_query_term_counts))
                    )

            row_time = row.get("time")
            if pd.notna(row_time):
                metrics["iteration_timeline_rows"].append(
                    {
                        "month": row_time.to_period("M").to_timestamp(),
                        "iteration_bucket": _timeline_iteration_bucket(num_iterations),
                    }
                )

        return metrics

    def _sorted_cdf(values):
        sorted_values = np.sort(np.asarray(values, dtype=float))
        cdf_values = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
        return sorted_values, cdf_values

    platform_metrics = {}
    platform_data_sources = {}
    for platform, _ in platform_configs:
        platform_df, source_path = _load_platform_df(platform)
        if platform_df is None:
            continue
        if "web_queries" not in platform_df.columns:
            print(f"Column `web_queries` not found for `{platform}`.")
            continue

        retrieved_url_count_lookup = _load_retrieved_url_count_lookup(platform)
        metrics = _collect_platform_metrics(
            platform_df,
            platform,
            retrieved_url_count_lookup,
        )
        if (
            len(metrics["total_queries_per_prompt"]) == 0
            or len(metrics["iterations_per_prompt"]) == 0
        ):
            print(f"No valid rows with web queries for `{platform}`.")
            continue

        platform_metrics[platform] = metrics
        platform_data_sources[platform] = source_path

    if len(platform_metrics) == 0:
        print("No valid rows found for any platform in query-term CDF plots.")
        return {}

    query_complexity_output_dir = f"{OUTPUT_PATH}/{CONF}/query_complexity"
    os.makedirs(query_complexity_output_dir, exist_ok=True)
    palette = px.colors.qualitative.Plotly

    def _plot_cdf_by_platform(
        metric_key,
        *,
        value_col,
        xaxis_title,
        file_name,
        hover_label,
        x_fmt=".2f",
        xaxis_config=None,
    ):
        def _format_tick(value):
            if abs(value - round(value)) < 1e-9:
                return str(int(round(value)))
            return f"{value:g}"

        def _nice_tick_step(span):
            if span <= 0:
                return 1.0
            raw = span / 5.0
            power = 10 ** np.floor(np.log10(raw))
            ratio = raw / power
            if ratio <= 1:
                nice = 1
            elif ratio <= 2:
                nice = 2
            elif ratio <= 5:
                nice = 5
            else:
                nice = 10
            return float(nice * power)

        fig = go.Figure()
        points_by_platform = {}

        for idx, (platform, display_name) in enumerate(platform_configs):
            if platform not in platform_metrics:
                continue

            values = platform_metrics[platform][metric_key]
            if len(values) == 0:
                continue

            sorted_values, cdf_values = _sorted_cdf(values)
            points_by_platform[platform] = pd.DataFrame(
                {value_col: sorted_values, "cdf": cdf_values}
            ).to_dict(orient="records")

            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=sorted_values,
                    y=cdf_values,
                    mode="lines",
                    name=display_name,
                    line=dict(width=4, color=color),
                    hovertemplate=(
                        f"{hover_label}: %{{x:{x_fmt}}}<br>"
                        "CDF: %{y:.3f}"
                        f"<extra>{display_name}</extra>"
                    ),
                )
            )

        layout_kwargs = {
            "xaxis_title": xaxis_title,
            "yaxis_title": "Cumulative Probability",
            "yaxis": dict(range=[0, 1]),
            "margin": dict(t=5),
        }
        if xaxis_config is not None:
            xaxis_settings = dict(xaxis_config)
            range_values = xaxis_settings.get("range")
            if (
                isinstance(range_values, (list, tuple))
                and len(range_values) == 2
            ):
                try:
                    range_start = float(range_values[0])
                    range_end = float(range_values[1])
                    if np.isfinite(range_start) and np.isfinite(range_end) and range_end > range_start:
                        tick_step = _nice_tick_step(range_end - range_start)
                        tick_values = np.arange(
                            range_start, range_end + (tick_step * 0.5), tick_step
                        )
                        if len(tick_values) == 0:
                            tick_values = np.array([range_start, range_end], dtype=float)
                        if tick_values[-1] < (range_end - 1e-9):
                            tick_values = np.append(tick_values, range_end)
                        else:
                            tick_values[-1] = range_end
                        tick_values = np.unique(np.round(tick_values, 10))
                        tick_text = [_format_tick(v) for v in tick_values]
                        if len(tick_text) > 0:
                            tick_text[-1] = f"{tick_text[-1]}+"
                        xaxis_settings.update(
                            {
                                "tickmode": "array",
                                "tickvals": tick_values.tolist(),
                                "ticktext": tick_text,
                            }
                        )
                except (TypeError, ValueError):
                    pass
            layout_kwargs["xaxis"] = xaxis_settings
        fig.update_layout(**layout_kwargs)

        fig = with_paper_style(fig, config=styler(26, 26), legend_pos=(0.8, 1.3))
        fig.write_image(f"{query_complexity_output_dir}/{file_name}.pdf", format="pdf")
        return points_by_platform

    web_file_name = "web_query_terms_cdf"
    user_file_name = "user_prompt_terms_cdf"
    avg_query_terms_per_prompt_file_name = "avg_web_query_terms_per_prompt_cdf"
    total_query_terms_per_prompt_file_name = "total_web_query_terms_per_prompt_cdf"
    total_queries_file_name = "total_web_queries_per_prompt_cdf"
    retrieved_urls_per_prompt_file_name = "retrieved_urls_per_prompt_cdf"
    retrieved_urls_per_web_query_file_name = "retrieved_urls_per_web_query_cdf"
    iterations_file_name = "iterations_per_prompt_cdf"
    timeline_file_name = "web_prompts_by_iteration_over_time"

    web_query_cdf_points_by_platform = _plot_cdf_by_platform(
        "terms_per_query",
        value_col="web_query_terms",
        xaxis_title="Number of Terms",
        file_name=web_file_name,
        hover_label="Web query terms",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    user_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "terms_per_prompt",
        value_col="user_prompt_terms",
        xaxis_title="Number of Terms",
        file_name=user_file_name,
        hover_label="User prompt terms",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    avg_web_query_terms_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "avg_query_terms_per_prompt",
        value_col="avg_web_query_terms_per_prompt",
        xaxis_title="Average Number of Web Query Terms Per User Prompt",
        file_name=avg_query_terms_per_prompt_file_name,
        hover_label="Avg web query terms per prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    total_web_query_terms_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "total_query_terms_per_prompt",
        value_col="total_web_query_terms_per_prompt",
        xaxis_title="Total Number of Web Query Terms Per User Prompt",
        file_name=total_query_terms_per_prompt_file_name,
        hover_label="Total web query terms per prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    total_web_queries_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "total_queries_per_prompt",
        value_col="total_web_queries_per_prompt",
        xaxis_title="Number of Web Queries Per User Prompt",
        file_name=total_queries_file_name,
        hover_label="Total web queries per prompt",
        x_fmt=".0f",
        xaxis_config=dict(range=[0, 10]),
    )
    retrieved_urls_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "retrieved_urls_per_prompt",
        value_col="retrieved_urls_per_prompt",
        xaxis_title="#Search Result URLs Per User Prompt",
        file_name=retrieved_urls_per_prompt_file_name,
        hover_label="#Search Result URLs per user prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    retrieved_urls_per_web_query_cdf_points_by_platform = _plot_cdf_by_platform(
        "retrieved_urls_per_web_query",
        value_col="retrieved_urls_per_web_query",
        xaxis_title="#Search Result URLs Per Web Query",
        file_name=retrieved_urls_per_web_query_file_name,
        hover_label="Retrieved URLs per web query",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    iterations_per_prompt_cdf_points_by_platform = _plot_cdf_by_platform(
        "iterations_per_prompt",
        value_col="iterations_per_prompt",
        xaxis_title="Number of Iterations Per User Prompt",
        file_name=iterations_file_name,
        hover_label="Iterations per prompt",
        x_fmt=".0f",
        xaxis_config=dict(range=[0, 10]),
    )

    timeline_plot_files_by_platform = {}
    timeline_points_by_platform = {}
    for platform, display_name in platform_configs:
        if platform not in platform_metrics:
            continue

        iteration_timeline_df = pd.DataFrame(
            platform_metrics[platform]["iteration_timeline_rows"]
        )
        if len(iteration_timeline_df) == 0:
            continue

        monthly_iteration_counts = (
            iteration_timeline_df.groupby(["month", "iteration_bucket"])
            .size()
            .reset_index(name="num_prompts")
        )
        monthly_iteration_counts["iteration_bucket"] = pd.Categorical(
            monthly_iteration_counts["iteration_bucket"],
            categories=iteration_bucket_order,
            ordered=True,
        )
        months = sorted(monthly_iteration_counts["month"].dropna().unique().tolist())
        full_index = pd.MultiIndex.from_product(
            [months, iteration_bucket_order],
            names=["month", "iteration_bucket"],
        )
        monthly_iteration_counts = (
            monthly_iteration_counts.set_index(["month", "iteration_bucket"])
            .reindex(full_index, fill_value=0)
            .reset_index()
            .sort_values(["month", "iteration_bucket"])
        )
        monthly_iteration_counts["month_total_prompts"] = monthly_iteration_counts.groupby(
            "month"
        )["num_prompts"].transform("sum")
        monthly_iteration_counts["pct_prompts"] = np.where(
            monthly_iteration_counts["month_total_prompts"] > 0,
            (monthly_iteration_counts["num_prompts"] * 100.0)
            / monthly_iteration_counts["month_total_prompts"],
            0.0,
        )

        timeline_fig = go.Figure()
        iteration_bucket_display = {
            "1": "1 iteration",
            "2": "2 iterations",
            "3+": "3+ iterations",
        }
        for iteration_bucket in iteration_bucket_order:
            subset = monthly_iteration_counts[
                monthly_iteration_counts["iteration_bucket"] == iteration_bucket
            ].sort_values("month")
            customdata = np.column_stack(
                (
                    subset["num_prompts"].to_numpy(dtype=float),
                    subset["month_total_prompts"].to_numpy(dtype=float),
                )
            )
            timeline_fig.add_trace(
                go.Scatter(
                    x=subset["month"],
                    y=subset["pct_prompts"],
                    mode="lines+markers",
                    name=iteration_bucket_display.get(iteration_bucket, iteration_bucket),
                    customdata=customdata,
                    hovertemplate=(
                        "Month: %{x|%b %Y}<br>"
                        "Iteration group: %{fullData.name}<br>"
                        "Share of web-search prompts: %{y:.1f}%<br>"
                        "Prompts in group: %{customdata[0]:.0f}<br>"
                        "Total web-search prompts: %{customdata[1]:.0f}<extra></extra>"
                    ),
                )
            )

        timeline_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Share of User Prompts",
            title=f"User Prompts by Iteration Bucket Over Time ({display_name})",
            xaxis=dict(
                tickmode="linear",
                dtick="M1",
                tickformat="%b %Y",
                tickangle=-45,
            ),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            margin=dict(b=90),
        )
        platform_timeline_file_name = f"{timeline_file_name}_{platform}"
        timeline_fig = with_paper_style(timeline_fig, config=styler(18, 16))
        timeline_fig.write_image(
            f"{query_complexity_output_dir}/{platform_timeline_file_name}.pdf", format="pdf"
        )

        timeline_records = monthly_iteration_counts.copy()
        timeline_records["month"] = timeline_records["month"].dt.strftime("%Y-%m")
        timeline_records["iteration_bucket"] = timeline_records["iteration_bucket"].astype(
            str
        )
        timeline_points_by_platform[platform] = timeline_records.to_dict(orient="records")
        timeline_plot_files_by_platform[platform] = platform_timeline_file_name

    summary_by_platform = {}
    def _safe_stat(values, reducer):
        if len(values) == 0:
            return None
        return float(reducer(values))

    for platform, metrics in platform_metrics.items():
        summary_by_platform[platform] = {
            "num_web_queries": int(len(metrics["terms_per_query"])),
            "num_prompts": int(len(metrics["terms_per_prompt"])),
            "mean_web_query_terms": _safe_stat(
                metrics["terms_per_query"], np.mean
            ),
            "median_web_query_terms": _safe_stat(
                metrics["terms_per_query"], np.median
            ),
            "mean_user_prompt_terms": _safe_stat(
                metrics["terms_per_prompt"], np.mean
            ),
            "median_user_prompt_terms": _safe_stat(
                metrics["terms_per_prompt"], np.median
            ),
            "mean_avg_web_query_terms_per_prompt": _safe_stat(
                metrics["avg_query_terms_per_prompt"], np.mean
            ),
            "median_avg_web_query_terms_per_prompt": _safe_stat(
                metrics["avg_query_terms_per_prompt"], np.median
            ),
            "mean_total_web_query_terms_per_prompt": _safe_stat(
                metrics["total_query_terms_per_prompt"], np.mean
            ),
            "median_total_web_query_terms_per_prompt": _safe_stat(
                metrics["total_query_terms_per_prompt"], np.median
            ),
            "mean_total_web_queries_per_prompt": float(
                np.mean(metrics["total_queries_per_prompt"])
            ),
            "median_total_web_queries_per_prompt": float(
                np.median(metrics["total_queries_per_prompt"])
            ),
            "mean_retrieved_urls_per_prompt": _safe_stat(
                metrics["retrieved_urls_per_prompt"], np.mean
            ),
            "median_retrieved_urls_per_prompt": _safe_stat(
                metrics["retrieved_urls_per_prompt"], np.median
            ),
            "mean_iterations_per_prompt": float(np.mean(metrics["iterations_per_prompt"])),
            "median_iterations_per_prompt": float(
                np.median(metrics["iterations_per_prompt"])
            ),
        }

    primary_platform = "openai"
    if primary_platform not in platform_metrics:
        primary_platform = next(iter(platform_metrics.keys()))
    primary_summary = summary_by_platform[primary_platform]
    primary_timeline_points = timeline_points_by_platform.get(primary_platform, [])
    primary_timeline_file = timeline_plot_files_by_platform.get(primary_platform)

    return {
        "platforms_plotted": list(platform_metrics.keys()),
        "platform_data_sources": platform_data_sources,
        "web_query_cdf_points_by_platform": web_query_cdf_points_by_platform,
        "user_prompt_cdf_points_by_platform": user_prompt_cdf_points_by_platform,
        "avg_web_query_terms_per_prompt_cdf_points_by_platform": avg_web_query_terms_per_prompt_cdf_points_by_platform,
        "total_web_query_terms_per_prompt_cdf_points_by_platform": total_web_query_terms_per_prompt_cdf_points_by_platform,
        "total_web_queries_per_prompt_cdf_points_by_platform": total_web_queries_per_prompt_cdf_points_by_platform,
        "retrieved_urls_per_prompt_cdf_points_by_platform": retrieved_urls_per_prompt_cdf_points_by_platform,
        "retrieved_urls_per_web_query_cdf_points_by_platform": retrieved_urls_per_web_query_cdf_points_by_platform,
        "iterations_per_prompt_cdf_points_by_platform": iterations_per_prompt_cdf_points_by_platform,
        "web_query_cdf_points": web_query_cdf_points_by_platform.get(primary_platform, []),
        "user_prompt_cdf_points": user_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "avg_web_query_terms_per_prompt_cdf_points": avg_web_query_terms_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "total_web_query_terms_per_prompt_cdf_points": total_web_query_terms_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "total_web_queries_per_prompt_cdf_points": total_web_queries_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "retrieved_urls_per_prompt_cdf_points": retrieved_urls_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "retrieved_urls_per_web_query_cdf_points": retrieved_urls_per_web_query_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "iterations_per_prompt_cdf_points": iterations_per_prompt_cdf_points_by_platform.get(
            primary_platform, []
        ),
        "web_prompts_by_iteration_over_time_points": primary_timeline_points,
        "web_prompts_by_iteration_over_time_points_by_platform": timeline_points_by_platform,
        "summary_by_platform": summary_by_platform,
        "num_web_queries": primary_summary["num_web_queries"],
        "num_prompts": primary_summary["num_prompts"],
        "web_query_plot_file": web_file_name,
        "user_prompt_plot_file": user_file_name,
        "avg_web_query_terms_per_prompt_plot_file": avg_query_terms_per_prompt_file_name,
        "total_web_query_terms_per_prompt_plot_file": total_query_terms_per_prompt_file_name,
        "total_web_queries_per_prompt_plot_file": total_queries_file_name,
        "retrieved_urls_per_prompt_plot_file": retrieved_urls_per_prompt_file_name,
        "retrieved_urls_per_web_query_plot_file": retrieved_urls_per_web_query_file_name,
        "iterations_per_prompt_plot_file": iterations_file_name,
        "web_prompts_by_iteration_over_time_plot_file": primary_timeline_file,
        "web_prompts_by_iteration_over_time_plot_files_by_platform": timeline_plot_files_by_platform,
        "timeline_platform": primary_platform,
        "mean_web_query_terms": primary_summary["mean_web_query_terms"],
        "median_web_query_terms": primary_summary["median_web_query_terms"],
        "mean_user_prompt_terms": primary_summary["mean_user_prompt_terms"],
        "median_user_prompt_terms": primary_summary["median_user_prompt_terms"],
        "mean_avg_web_query_terms_per_prompt": primary_summary[
            "mean_avg_web_query_terms_per_prompt"
        ],
        "median_avg_web_query_terms_per_prompt": primary_summary[
            "median_avg_web_query_terms_per_prompt"
        ],
        "mean_total_web_query_terms_per_prompt": primary_summary[
            "mean_total_web_query_terms_per_prompt"
        ],
        "median_total_web_query_terms_per_prompt": primary_summary[
            "median_total_web_query_terms_per_prompt"
        ],
        "mean_total_web_queries_per_prompt": primary_summary[
            "mean_total_web_queries_per_prompt"
        ],
        "median_total_web_queries_per_prompt": primary_summary[
            "median_total_web_queries_per_prompt"
        ],
        "mean_iterations_per_prompt": primary_summary["mean_iterations_per_prompt"],
        "median_iterations_per_prompt": primary_summary["median_iterations_per_prompt"],
    }


def _safe_json_value(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return default
    return value


def _detect_language_safe(text):
    text = str(text or "").strip()
    if not text:
        return ""
    try:
        return detect(text)
    except Exception:
        return ""


def detect_user_and_web_query_languages(
    input_path=None,
    output_stem="query_reformulations_user_and_web_query_languages",
    platform="chatgpt",
):
    input_path = input_path or (
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    if input_path.endswith(".pkl"):
        df = pd.read_pickle(input_path).copy()
    else:
        df = pd.read_csv(input_path).copy()

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        time_value = row.get("time")
        if time_value is not None:
            try:
                if pd.isna(time_value):
                    time_value = None
            except (TypeError, ValueError):
                pass
        if time_value is not None:
            time_value = str(time_value)

        base_record = {
            "user_id": row.get("user_id"),
            "conv_id": row.get("conv_id"),
            "turn_id": row.get("turn_id"),
            "topic": row.get("topic"),
            "row_language": row.get("language"),
            "time": time_value,
        }

        web_query_groups = _safe_json_value(row.get("web_queries"), [])
        if not isinstance(web_query_groups, list):
            web_query_groups = []

        web_query_records = []
        for iteration_idx, query_group in enumerate(web_query_groups, start=1):
            if not isinstance(query_group, list):
                query_group = [query_group]
            for query_idx, web_query in enumerate(query_group, start=1):
                if not isinstance(web_query, str):
                    continue
                web_query = web_query.strip()
                if not web_query:
                    continue
                web_query_records.append(
                    {
                        **base_record,
                        "text_source": "web_query",
                        "iteration_idx": iteration_idx,
                        "query_idx": query_idx,
                        "query_id": f"{iteration_idx}.{query_idx}",
                        "text": web_query,
                        "detected_language": _detect_language_safe(web_query),
                        "text_length": len(web_query),
                        "num_words": len(web_query.split()),
                    }
                )

        if not web_query_records:
            continue

        user_query = str(_row_latest_user_query(row) or "").strip()
        if user_query:
            records.append(
                {
                    **base_record,
                    "text_source": "user_prompt",
                    "iteration_idx": 0,
                    "query_idx": 0,
                    "query_id": "U",
                    "text": user_query,
                    "detected_language": _detect_language_safe(user_query),
                    "text_length": len(user_query),
                    "num_words": len(user_query.split()),
                }
            )
        records.extend(web_query_records)

    records_df = pd.DataFrame(records)
    _save_dataframe(records_df, output_stem, platform=platform)
    to_json(
        records_df.to_dict(orient="records"),
        f"{OUTPUT_PATH}/{platform}/metadata/{output_stem}.json",
    )

    if len(records_df) == 0:
        print("No user/web-query language rows found.")
        return records_df

    summary_df = (
        records_df.groupby(["text_source", "iteration_idx", "detected_language"])
        .size()
        .reset_index(name="count")
        .sort_values(
            ["text_source", "iteration_idx", "count"],
            ascending=[True, True, False],
        )
    )
    summary_stem = f"{output_stem}_summary"
    _save_dataframe(summary_df, summary_stem, platform=platform)
    to_json(
        summary_df.to_dict(orient="records"),
        f"{OUTPUT_PATH}/{platform}/metadata/{summary_stem}.json",
    )

    print("\nDetected language summary:")
    print(summary_df.to_string(index=False))
    return records_df


def plot_user_and_web_query_language_patterns(
    input_stem="query_reformulations_user_and_web_query_languages",
    output_file_name="query_reformulations_user_and_web_query_language_patterns",
    top_n_pairs=10,
    platform="chatgpt",
):
    input_pkl_path = f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.pkl"
    input_csv_path = f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.csv"
    if os.path.exists(input_pkl_path):
        df = pd.read_pickle(input_pkl_path).copy()
    else:
        df = pd.read_csv(input_csv_path).copy()

    if df.empty:
        print("No language detection rows found.")
        return pd.DataFrame()

    sample_key_cols = ["user_id", "conv_id", "turn_id"]
    missing_cols = [col for col in sample_key_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing sample key columns: {missing_cols}")
        return pd.DataFrame()

    df["detected_language"] = (
        df["detected_language"].fillna("").astype(str).str.strip()
    )
    user_df = df[df["text_source"] == "user_prompt"].copy()
    web_df = df[df["text_source"] == "web_query"].copy()
    web_df = web_df[web_df["detected_language"] != ""].copy()

    if web_df.empty:
        print("No web-query language rows found.")
        return pd.DataFrame()

    sample_web_language_counts = (
        web_df.groupby(sample_key_cols)["detected_language"]
        .nunique()
        .reset_index(name="num_web_query_languages")
    )
    sample_web_language_counts["language_count_bucket"] = sample_web_language_counts[
        "num_web_query_languages"
    ].apply(
        lambda x: (
            "1 language"
            if x == 1
            else "2 languages"
            if x == 2
            else "3 languages"
            if x == 3
            else "4+ languages"
        )
    )
    language_count_summary = (
        sample_web_language_counts["language_count_bucket"]
        .value_counts()
        .rename_axis("language_count_bucket")
        .reset_index(name="num_samples")
    )
    language_count_summary["percentage"] = (
        language_count_summary["num_samples"]
        / language_count_summary["num_samples"].sum()
    )
    bucket_order = ["1 language", "2 languages", "3 languages", "4+ languages"]
    language_count_summary["language_count_bucket"] = pd.Categorical(
        language_count_summary["language_count_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    language_count_summary = language_count_summary.sort_values(
        "language_count_bucket"
    )

    web_language_counts = (
        web_df.groupby(sample_key_cols + ["detected_language"])
        .size()
        .reset_index(name="count")
        .sort_values(
            sample_key_cols + ["count", "detected_language"],
            ascending=[True, True, True, False, True],
        )
    )
    dominant_web_language = (
        web_language_counts.sort_values(
            sample_key_cols + ["count", "detected_language"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(subset=sample_key_cols, keep="first")
        .rename(columns={"detected_language": "dominant_web_query_language"})
    )

    user_language = (
        user_df.sort_values(sample_key_cols)
        .drop_duplicates(subset=sample_key_cols, keep="first")
        [sample_key_cols + ["detected_language"]]
        .rename(columns={"detected_language": "user_prompt_language"})
    )
    language_pairs = dominant_web_language.merge(
        user_language,
        on=sample_key_cols,
        how="inner",
    )
    language_pairs = language_pairs[
        (language_pairs["user_prompt_language"] != "")
        & (language_pairs["dominant_web_query_language"] != "")
    ].copy()
    language_pairs["language_pair"] = (
        language_pairs["user_prompt_language"]
        + " → "
        + language_pairs["dominant_web_query_language"]
    )

    pair_summary = (
        language_pairs["language_pair"]
        .value_counts()
        .rename_axis("language_pair")
        .reset_index(name="num_samples")
        .head(top_n_pairs)
    )
    pair_summary["percentage"] = pair_summary["num_samples"] / len(language_pairs)

    os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
    language_count_summary.to_csv(
        f"{OUTPUT_PATH}/{platform}/metadata/{output_file_name}_web_language_count_summary.csv",
        index=False,
    )
    pair_summary.to_csv(
        f"{OUTPUT_PATH}/{platform}/metadata/{output_file_name}_top_language_pairs.csv",
        index=False,
    )
    language_pairs.to_csv(
        f"{OUTPUT_PATH}/{platform}/metadata/{output_file_name}_language_pairs.csv",
        index=False,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Web Query Language Diversity",
            f"Top {top_n_pairs} User → Web Query Language Pairs",
        ],
        horizontal_spacing=0.18,
    )
    fig.update_annotations(font_size=22)
    fig.add_trace(
        go.Bar(
            x=language_count_summary["language_count_bucket"].astype(str),
            y=language_count_summary["percentage"],
            customdata=language_count_summary["num_samples"],
            showlegend=False,
            hovertemplate=(
                "Bucket: %{x}<br>"
                "Share: %{y:.1%}<br>"
                "Samples: %{customdata}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=pair_summary["percentage"],
            y=pair_summary["language_pair"],
            orientation="h",
            customdata=pair_summary["num_samples"],
            showlegend=False,
            hovertemplate=(
                "Pair: %{y}<br>"
                "Share: %{x:.1%}<br>"
                "Samples: %{customdata}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Share of samples", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="Share of samples", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(
        width=1150,
        height=560,
        margin=dict(t=90, b=80, l=80, r=40),
    )
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=None)
    fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{output_file_name}.pdf", format="pdf")

    print("\nWeb query language-count summary:")
    print(language_count_summary.to_string(index=False))
    print(f"\nTop {top_n_pairs} user to dominant-web-query language pairs:")
    print(pair_summary.to_string(index=False))
    return {
        "language_count_summary": language_count_summary,
        "pair_summary": pair_summary,
        "language_pairs": language_pairs,
    }


def _get_loop_items(loop_items, loop_idx):
    if not isinstance(loop_items, list) or loop_idx >= len(loop_items):
        return []
    items = loop_items[loop_idx]
    if isinstance(items, list):
        return items
    return [items]


def _detect_web_query_token_sources(
    latest_user_query,
    web_query_groups,
    previous_user_or_assistant_history=None,
    all_sources=None,
    all_memories=None,
):
    latest_user_query_words = set(preprocess_text_in_chunks(latest_user_query or ""))
    previous_history_words = set(
        preprocess_texts(previous_user_or_assistant_history or [])
    )

    web_query_groups = _safe_json_value(web_query_groups, [])
    all_sources = _safe_json_value(all_sources, [])
    all_memories = _safe_json_value(all_memories, [])

    new_words_from_user_queries = []
    new_words_from_assistant_queries = []
    new_words_from_thoughts = []
    new_words_from_sources = []
    new_words_from_memories = []
    new_words_from_pk = []

    for loop_idx, system_queries in enumerate(web_query_groups):
        if not isinstance(system_queries, list):
            system_queries = [system_queries]

        sources_w = set(preprocess_texts(_get_loop_items(all_sources, loop_idx)))
        memories_w = set(preprocess_texts(_get_loop_items(all_memories, loop_idx)))

        for system_query in system_queries:
            for word in preprocess_text_in_chunks(system_query):
                if word in latest_user_query_words:
                    new_words_from_user_queries.append(word)
                elif word in previous_history_words:
                    # Backward-compatible column name: this includes previous
                    # user messages as well as assistant messages.
                    new_words_from_assistant_queries.append(word)
                elif word in memories_w:
                    new_words_from_memories.append(word)
                elif word in sources_w:
                    new_words_from_sources.append(word)
                else:
                    new_words_from_pk.append(word)

    return {
        "all_new_words_from_user_queries": list(set(new_words_from_user_queries)),
        "all_new_words_from_assistant_queries": list(
            set(new_words_from_assistant_queries)
        ),
        "all_new_words_from_sources": list(set(new_words_from_sources)),
        "all_new_words_from_thoughts": list(set(new_words_from_thoughts)),
        "all_new_words_from_memories": list(set(new_words_from_memories)),
        "all_new_words_from_pk": list(set(new_words_from_pk)),
    }


def _add_web_query_token_source_columns(df):
    token_source_rows = {
        "all_new_words_from_user_queries": [],
        "all_new_words_from_assistant_queries": [],
        "all_new_words_from_sources": [],
        "all_new_words_from_thoughts": [],
        "all_new_words_from_memories": [],
        "all_new_words_from_pk": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_msg_history = _safe_json_value(row.get("user_msg_history"), [])
        assistant_msg_history = _safe_json_value(row.get("assistant_msg_history"), [])
        if isinstance(user_msg_history, str):
            user_msg_history = [user_msg_history]
        if isinstance(assistant_msg_history, str):
            assistant_msg_history = [assistant_msg_history]

        latest_user_query = user_msg_history[-1] if user_msg_history else ""
        previous_history = user_msg_history[:-1] + assistant_msg_history

        token_sources = _detect_web_query_token_sources(
            latest_user_query=latest_user_query,
            web_query_groups=row.get("web_queries"),
            previous_user_or_assistant_history=previous_history,
            all_sources=row.get("sources"),
            all_memories=row.get("memories"),
        )
        for col in token_source_rows:
            token_source_rows[col].append(token_sources[col])

    df = df.copy()
    for col, values in token_source_rows.items():
        df[col] = values
    return df


def _save_dataframe(df, output_stem, platform="chatgpt"):
    csv_path = f"{OUTPUT_PATH}/{platform}/metadata/{output_stem}.csv"
    pkl_path = f"{OUTPUT_PATH}/{platform}/metadata/{output_stem}.pkl"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)


def _filter_query_reformulation_df(df):
    filtered_rows = []
    num_wq = 0

    for _, row in df.iterrows():
        if row["conv_starter"] != 1:
            continue

        if row["language"] != "en":
            continue

        user_msg_history = row["user_msg_history"]
        if len(user_msg_history) != 1:
            continue

        user_query = user_msg_history[-1]
        if not user_query:
            continue

        web_queries = json.loads(row["web_queries"])
        all_web_queries = [q for qs in web_queries for q in qs]
        num_wq += len(all_web_queries)

        # if not all_web_queries:
        #     continue

        filtered_rows.append(row.to_dict())

    print(len(filtered_rows), num_wq)
    return pd.DataFrame(filtered_rows)


def _row_latest_user_query(row):
    user_msg_history = _safe_json_value(row.get("user_msg_history"), [])
    if isinstance(user_msg_history, str):
        user_msg_history = [user_msg_history]
    if user_msg_history:
        return user_msg_history[-1]
    return row.get("user_prompt") or row.get("user_query") or ""


def _base_query_record(row):
    return {
        "result_key": row.get("result_key"),
        "sample_source": row.get("sample_source"),
        "user_id": row.get("user_id"),
        "conv_id": row.get("conv_id"),
        "turn_id": row.get("turn_id"),
        "topic": row.get("topic"),
        "language": row.get("language"),
        "invivo_model": row.get("invivo_model"),
        "replay_model": row.get("replay_model"),
        "response_mode": row.get("response_mode"),
    }


def _save_query_eval_records(records, output_stem, platform="chatgpt"):
    results_df = pd.DataFrame(records)
    _save_dataframe(results_df, output_stem, platform=platform)
    to_json(records, f"{OUTPUT_PATH}/{platform}/metadata/{output_stem}.json")
    return results_df


def _load_dataframe_from_candidates(candidate_paths):
    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            continue
        try:
            if candidate_path.endswith(".pkl"):
                return pd.read_pickle(candidate_path).copy(), candidate_path
            return pd.read_csv(candidate_path).copy(), candidate_path
        except Exception as e:
            print(f"Failed to load `{candidate_path}`: {e}")
    return None, None

def _build_query_specificity_stage_df(df, max_web_stage_bucket=3):
    specificity_dimensions = [
        "temporal",
        "geographic",
        "entity",
        "numeric",
    ]
    specificity_label_map = {
        "temporal": "Temporal",
        "geographic": "Geographic",
        "entity": "Entity",
        "numeric": "Numeric",
    }
    score_order = [1, 2, 3, 4, 5]
    stage_dimension_score_counts = {}

    def _bucket_stage_idx(stage_idx):
        if stage_idx <= 0:
            return 0
        return min(stage_idx, max_web_stage_bucket)

    def _add_score(stage_idx, dimension, score):
        dimension = str(dimension or "").strip().lower()
        if dimension not in specificity_label_map:
            return
        try:
            score = int(score)
        except (TypeError, ValueError):
            return
        if score not in score_order:
            return

        stage_idx = _bucket_stage_idx(stage_idx)
        stage_dimension_score_counts.setdefault(stage_idx, {})
        stage_dimension_score_counts[stage_idx].setdefault(dimension, Counter())
        stage_dimension_score_counts[stage_idx][dimension][score] += 1

    def _add_specificity_dict(stage_idx, specificity_dict):
        if not isinstance(specificity_dict, dict):
            return
        for dimension in specificity_dimensions:
            judgment = specificity_dict.get(dimension, {})
            if not isinstance(judgment, dict):
                continue
            _add_score(stage_idx, dimension, judgment.get("score"))

    for _, row in df.iterrows():
        user_query_specificity = _safe_json_value(
            row.get("user_query_specificity", "{}"),
            {},
        )
        _add_specificity_dict(0, user_query_specificity)

        web_query_specificity_info = _safe_json_value(
            row.get("web_query_specificity_info", "[]"),
            [],
        )

        if isinstance(web_query_specificity_info, list):
            for item in web_query_specificity_info:
                if not isinstance(item, dict):
                    continue
                iteration_idx = item.get("iteration")
                try:
                    iteration_idx = int(iteration_idx)
                except (TypeError, ValueError):
                    continue
                _add_specificity_dict(
                    iteration_idx,
                    item.get("specificity", {}),
                )
        elif isinstance(web_query_specificity_info, dict):
            for item in web_query_specificity_info.values():
                if not isinstance(item, dict):
                    continue
                iteration_idx = item.get("iteration", 1)
                try:
                    iteration_idx = int(iteration_idx)
                except (TypeError, ValueError):
                    iteration_idx = 1
                specificity_dict = item.get("specificity", item)
                _add_specificity_dict(iteration_idx, specificity_dict)

    rows = []
    stage_indices = sorted(stage_dimension_score_counts.keys())
    for stage_idx in stage_indices:
        if stage_idx == 0:
            stage_label = "User"
        elif stage_idx == max_web_stage_bucket:
            stage_label = f"Iter. {max_web_stage_bucket}+"
        else:
            stage_label = f"Iter. {stage_idx}"

        for dimension in specificity_dimensions:
            score_counter = stage_dimension_score_counts.get(stage_idx, {}).get(
                dimension,
                Counter(),
            )
            total = sum(score_counter.values())
            for score in score_order:
                count = score_counter.get(score, 0)
                rows.append(
                    {
                        "stage_idx": stage_idx,
                        "stage_label": stage_label,
                        "dimension": dimension,
                        "dimension_display": specificity_label_map[dimension],
                        "score": score,
                        "score_display": f"Score {score}",
                        "count": count,
                        "total": total,
                        "rate": (count / total) if total > 0 else 0.0,
                    }
                )

    return pd.DataFrame(rows)


def plot_query_specificity_distribution_by_iteration(
    input_stem="query_reformulations_query_specificity",
    output_file_name="query_specificity_distribution_by_iteration",
    platform="chatgpt",
):
    platform_configs = [
        ("openai", "ChatGPT"),
        ("claude", "Claude"),
        ("grok", "Grok"),
        ("deepseek", "DeepSeek"),
    ]

    def _specificity_score_vector(specificity_dict):
        if not isinstance(specificity_dict, dict):
            return {}
        scores = {}
        for dimension, judgment in specificity_dict.items():
            if not isinstance(judgment, dict):
                continue
            try:
                score = int(judgment.get("score"))
            except (TypeError, ValueError):
                continue
            scores[str(dimension).strip().lower()] = score
        return scores

    def _overall_specificity_direction(user_specificity_dict, query_specificity_dict):
        user_scores = _specificity_score_vector(user_specificity_dict)
        query_scores = _specificity_score_vector(query_specificity_dict)
        shared_dimensions = sorted(set(user_scores) & set(query_scores))
        if not shared_dimensions:
            return None
        deltas = [query_scores[dim] - user_scores[dim] for dim in shared_dimensions]
        if any(delta > 0 for delta in deltas):
            return 1
        if all(delta == 0 for delta in deltas):
            return 0
        return -1

    def _dimension_specificity_direction(
        user_specificity_dict,
        query_specificity_dict,
        dimension,
    ):
        user_scores = _specificity_score_vector(user_specificity_dict)
        query_scores = _specificity_score_vector(query_specificity_dict)
        dimension = str(dimension or "").strip().lower()
        if dimension not in user_scores or dimension not in query_scores:
            return None
        delta = query_scores[dimension] - user_scores[dimension]
        if delta > 0:
            return 1
        if delta == 0:
            return 0
        return 0

    def _platform_specificity_candidates(platform):
        if platform in {"openai", "chatgpt"}:
            return [
                f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.pkl",
                f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.csv",
            ]
        return [
            f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.pkl",
            f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.csv",
        ]

    input_pkl_path = f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.pkl"
    input_csv_path = f"{OUTPUT_PATH}/{platform}/metadata/{input_stem}.csv"
    if os.path.exists(input_pkl_path):
        df = pd.read_pickle(input_pkl_path).copy()
    else:
        df = pd.read_csv(input_csv_path).copy()

    plot_df = _build_query_specificity_stage_df(df)
    if plot_df.empty:
        print("No query specificity rows to plot.")
        return pd.DataFrame()

    os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
    plot_df.to_csv(
        f"{OUTPUT_PATH}/{platform}/metadata/{output_file_name}.csv",
        index=False,
    )

    dimensions = ["temporal", "geographic", "entity", "numeric"]
    dimension_titles = ["Temporal", "Geographic", "Entity", "Numeric"]
    score_order = [1, 2, 3, 4, 5]
    score_color_map = {
        1: "#4c78a8",
        2: "#72b7b2",
        3: "#f2cf5b",
        4: "#f58518",
        5: "#e45756",
    }

    def _write_specificity_distribution_plot(platform_plot_df, plot_base_name):
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_yaxes=True,
            subplot_titles=dimension_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.23,
        )
        fig.update_annotations(font_size=23)

        for dim_idx, dimension in enumerate(dimensions):
            row_idx = (dim_idx // 2) + 1
            col_idx = (dim_idx % 2) + 1
            dimension_df = platform_plot_df[
                platform_plot_df["dimension"] == dimension
            ].copy()
            stage_df = (
                dimension_df[["stage_idx", "stage_label"]]
                .drop_duplicates()
                .sort_values("stage_idx")
            )
            stage_labels = stage_df["stage_label"].tolist()

            for score in score_order:
                score_df = dimension_df[
                    dimension_df["score"] == score
                ].sort_values("stage_idx")
                if score_df.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=score_df["stage_label"],
                        y=score_df["rate"],
                        name=f"Score {score}",
                        legendgroup=f"score_{score}",
                        showlegend=dim_idx == 0,
                        marker_color=score_color_map[score],
                        customdata=score_df[["count", "total"]].values,
                        hovertemplate=(
                            "Stage: %{x}<br>"
                            "Score: %{fullData.name}<br>"
                            "Share: %{y:.1%}<br>"
                            "Count: %{customdata[0]} / %{customdata[1]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            fig.update_xaxes(
                title_text="",
                categoryorder="array",
                categoryarray=stage_labels,
                tickangle=0,
                tickfont=dict(size=18),
                row=row_idx,
                col=col_idx,
            )

        fig.update_layout(
            barmode="stack",
            margin=dict(t=90, b=100, l=100, r=30),
            legend_title="",
        )
        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1.0],
        )
        fig.add_annotation(
            x=-0.15,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Share",
            showarrow=False,
            textangle=-90,
            font=dict(size=20),
        )
        fig.add_annotation(
            x=0.5,
            y=-0.22,
            xref="paper",
            yref="paper",
            text="Query Formulation Iteration",
            showarrow=False,
            font=dict(size=20),
        )

        fig = with_paper_style(fig, config=styler(20, 18), legend_pos=(1, 1.25))
        fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{plot_base_name}.pdf", format="pdf")

    _write_specificity_distribution_plot(plot_df, output_file_name)

    specificity_stage_frames = [plot_df.assign(platform="openai", platform_display="ChatGPT")]
    specificity_source_paths = {"openai": input_pkl_path if os.path.exists(input_pkl_path) else input_csv_path}
    for platform, display_name in platform_configs:
        if platform in {"openai", "chatgpt"}:
            continue
        platform_df, source_path = _load_dataframe_from_candidates(
            _platform_specificity_candidates(platform)
        )
        if platform_df is None:
            print(f"No query specificity file found for `{platform}`.")
            continue
        platform_plot_df = _build_query_specificity_stage_df(platform_df)
        if platform_plot_df.empty:
            print(f"No query specificity rows to plot for `{platform}`.")
            continue
        platform_plot_df["platform"] = platform
        platform_plot_df["platform_display"] = display_name
        specificity_stage_frames.append(platform_plot_df)
        specificity_source_paths[platform] = source_path
        _write_specificity_distribution_plot(
            platform_plot_df,
            f"{output_file_name}_{platform}",
        )

    if len(specificity_stage_frames) > 1:
        combined_specificity_df = pd.concat(specificity_stage_frames, ignore_index=True)
        combined_specificity_df.to_csv(
            f"{OUTPUT_PATH}/{platform}/metadata/{output_file_name}_all_platforms.csv",
            index=False,
        )
        to_json(
            {
                "platforms_plotted": [
                    platform
                    for platform, _ in platform_configs
                    if platform in set(combined_specificity_df["platform"])
                ],
                "platform_data_sources": specificity_source_paths,
            },
            f"{OUTPUT_PATH}/{platform}/metadata/{output_file_name}_all_platforms_sources.json",
        )

    overall_rows = []
    dimension_direction_rows = []
    source_paths = {}
    dimensions = ["temporal", "geographic", "entity", "numeric"]
    dimension_line_color_map = {
        "temporal": "#636EFA",
        "geographic": "#EF553B",
        "entity": "#00CC96",
        "numeric": "#AB63FA",
        "overall": "#7F7F7F",
    }
    dimension_line_dash_map = {
        "temporal": "solid",
        "geographic": "solid",
        "entity": "solid",
        "numeric": "solid",
        "overall": "dash",
    }
    dimension_display_map = {
        "temporal": "Temporal",
        "geographic": "Geographic",
        "entity": "Entity",
        "numeric": "Numeric",
        "overall": "Overall",
    }
    for platform, display_name in platform_configs:
        platform_df, source_path = _load_dataframe_from_candidates(
            _platform_specificity_candidates(platform)
        )
        if platform_df is None:
            print(f"No query specificity file found for `{platform}`.")
            continue
        source_paths[platform] = source_path

        stage_direction_values = {}
        dimension_stage_direction_values = {
            dimension: {} for dimension in dimensions
        }
        for _, row in platform_df.iterrows():
            user_query_specificity = _safe_json_value(
                row.get("user_query_specificity", "{}"),
                {},
            )
            web_query_specificity_info = _safe_json_value(
                row.get("web_query_specificity_info", "[]"),
                [],
            )

            if isinstance(web_query_specificity_info, list):
                specificity_items = web_query_specificity_info
            elif isinstance(web_query_specificity_info, dict):
                specificity_items = list(web_query_specificity_info.values())
            else:
                specificity_items = []

            for item in specificity_items:
                if not isinstance(item, dict):
                    continue
                iteration_idx = item.get("iteration", 1)
                try:
                    iteration_idx = int(iteration_idx)
                except (TypeError, ValueError):
                    continue
                stage_idx = min(iteration_idx, 3) if iteration_idx > 0 else 0
                specificity_dict = item.get("specificity", item)
                direction = _overall_specificity_direction(
                    user_query_specificity,
                    specificity_dict,
                )
                if direction is not None:
                    stage_direction_values.setdefault(stage_idx, []).append(direction)
                for dimension in dimensions:
                    dimension_direction = _dimension_specificity_direction(
                        user_query_specificity,
                        specificity_dict,
                        dimension,
                    )
                    if dimension_direction is not None:
                        dimension_stage_direction_values[dimension].setdefault(
                            stage_idx,
                            [],
                        ).append(dimension_direction)

        for stage_idx in sorted(stage_direction_values):
            if stage_idx == 0:
                stage_label = "User"
            elif stage_idx >= 3:
                stage_label = "Iter. 3+"
            else:
                stage_label = f"Iter. {stage_idx}"
            values = stage_direction_values[stage_idx]
            mean_value = sum(values) / len(values) if values else None
            overall_rows.append(
                {
                    "platform": platform,
                    "platform_display": display_name,
                    "stage_idx": int(stage_idx),
                    "stage_label": stage_label,
                    "mean_overall_specificity_direction": float(mean_value)
                    if mean_value is not None
                    else None,
                    "percentage_overall_specificity_direction": float(
                        mean_value * 100.0
                    )
                    if mean_value is not None
                    else None,
                    "count": int(len(values)),
                }
            )

        for dimension in dimensions:
            for stage_idx in sorted(dimension_stage_direction_values[dimension]):
                if stage_idx == 0:
                    stage_label = "User"
                elif stage_idx >= 3:
                    stage_label = "Iter. 3+"
                else:
                    stage_label = f"Iter. {stage_idx}"
                values = dimension_stage_direction_values[dimension][stage_idx]
                mean_value = sum(values) / len(values) if values else None
                dimension_direction_rows.append(
                    {
                        "platform": platform,
                        "platform_display": display_name,
                        "dimension": dimension,
                        "dimension_display": dimension_display_map[dimension],
                        "stage_idx": int(stage_idx),
                        "stage_label": stage_label,
                        "mean_specificity_direction": float(mean_value)
                        if mean_value is not None
                        else None,
                        "percentage_specificity_direction": float(mean_value * 100.0)
                        if mean_value is not None
                        else None,
                        "count": int(len(values)),
                    }
                )

    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_output_file_name = (
            f"{output_file_name}_overall_specificity_direction"
        )
        overall_df.to_csv(
            f"{OUTPUT_PATH}/{platform}/metadata/{overall_output_file_name}.csv",
            index=False,
        )
        to_json(
            {
                "platforms_plotted": [
                    platform for platform, _ in platform_configs
                    if platform in set(overall_df["platform"])
                ],
                "platform_data_sources": source_paths,
            },
            f"{OUTPUT_PATH}/{platform}/metadata/{overall_output_file_name}_sources.json",
        )

        line_fig = go.Figure()
        platform_color_map = {
            "openai": "#636EFA",
            "claude": "#EF553B",
            "grok": "#00CC96",
            "deepseek": "#AB63FA",
        }
        platform_marker_map = {
            "openai": "circle",
            "claude": "star",
            "grok": "x",
            "deepseek": "diamond",
        }
        plotted_platforms = [
            (platform, display_name)
            for platform, display_name in platform_configs
            if platform in set(overall_df["platform"])
        ]
        for platform, display_name in plotted_platforms:
            platform_line_df = overall_df[
                overall_df["platform"] == platform
            ].sort_values("stage_idx")
            line_fig.add_trace(
                go.Scatter(
                    x=platform_line_df["stage_idx"],
                    y=platform_line_df["percentage_overall_specificity_direction"],
                    mode="lines+markers",
                    name=display_name,
                    line=dict(color=platform_color_map.get(platform)),
                    marker=dict(
                        size=16,
                        color=platform_color_map.get(platform),
                        symbol=platform_marker_map.get(platform, "circle"),
                    ),
                    customdata=platform_line_df[["count", "stage_label"]].values,
                    hovertemplate=(
                        "Platform: %{fullData.name}<br>"
                        "Stage: %{customdata[1]}<br>"
                        "Mean direction: %{y:.1f}%<br>"
                        "Queries: %{customdata[0]}<extra></extra>"
                    ),
                )
            )
        line_fig.update_layout(
            xaxis_title="Query Formulation Iteration",
            yaxis_title="Avg Specificity Increase (%)",
            margin=dict(t=30, b=80, l=80, r=30),
            legend_title="",
        )
        line_fig.update_xaxes(
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=[
                "User → Iter. 1",
                "Iter. 1 → Iter. 2",
                "Iter. 2+ → Iter. 3+",
            ],
            range=[0.8, 3.4],
            tickfont=dict(size=21),
        )
        line_fig.update_yaxes(ticksuffix="%")
        line_fig = with_paper_style(
            line_fig,
            config=styler(24, 24),
            # legend_pos=(0.9, 1.2),
        )
        line_fig.write_image(
            f"{OUTPUT_PATH}/{platform}/{CONF}/{overall_output_file_name}.pdf",
            format="pdf",
        )

    if dimension_direction_rows:
        dimension_direction_df = pd.DataFrame(dimension_direction_rows)
        dimension_output_file_name = (
            f"{output_file_name}_dimension_specificity_direction"
        )
        dimension_direction_df.to_csv(
            f"{OUTPUT_PATH}/{platform}/metadata/{dimension_output_file_name}.csv",
            index=False,
        )
        to_json(
            {
                "platforms_plotted": [
                    platform for platform, _ in platform_configs
                    if platform in set(dimension_direction_df["platform"])
                ],
                "platform_data_sources": source_paths,
            },
            f"{OUTPUT_PATH}/{platform}/metadata/{dimension_output_file_name}_sources.json",
        )

        if overall_rows:
            overall_df = pd.DataFrame(overall_rows)
        else:
            overall_df = pd.DataFrame()

        plotted_platforms = [
            (platform, display_name)
            for platform, display_name in platform_configs
            if platform in set(dimension_direction_df["platform"])
        ]
        for platform, display_name in plotted_platforms:
            platform_dimension_df = dimension_direction_df[
                dimension_direction_df["platform"] == platform
            ].copy()
            platform_overall_df = overall_df[
                overall_df["platform"] == platform
            ].copy()

            per_platform_line_fig = go.Figure()
            for dimension in dimensions:
                if dimension == "numeric":
                    continue
                dimension_df = platform_dimension_df[
                    platform_dimension_df["dimension"] == dimension
                ].sort_values("stage_idx")
                if dimension_df.empty:
                    continue
                per_platform_line_fig.add_trace(
                    go.Scatter(
                        x=dimension_df["stage_idx"],
                        y=dimension_df["percentage_specificity_direction"],
                        mode="lines+markers",
                        name=dimension_display_map[dimension],
                        line=dict(
                            color=dimension_line_color_map[dimension],
                            dash=dimension_line_dash_map[dimension],
                            width=4,
                        ),
                        marker=dict(
                            size=12,
                            color=dimension_line_color_map[dimension],
                        ),
                        customdata=dimension_df[["count", "stage_label"]].values,
                        hovertemplate=(
                            "Dimension: %{fullData.name}<br>"
                            "Stage: %{customdata[1]}<br>"
                            "Mean direction: %{y:.1f}%<br>"
                            "Queries: %{customdata[0]}<extra></extra>"
                        ),
                    )
                )

            # if not platform_overall_df.empty:
            #     platform_overall_df = platform_overall_df.sort_values("stage_idx")
            #     per_platform_line_fig.add_trace(
            #         go.Scatter(
            #             x=platform_overall_df["stage_idx"],
            #             y=platform_overall_df[
            #                 "percentage_overall_specificity_direction"
            #             ],
            #             mode="lines+markers",
            #             name=dimension_display_map["overall"],
            #             line=dict(
            #                 color=dimension_line_color_map["overall"],
            #                 dash=dimension_line_dash_map["overall"],
            #                 width=4,
            #             ),
            #             marker=dict(
            #                 size=12,
            #                 color=dimension_line_color_map["overall"],
            #             ),
            #             customdata=platform_overall_df[["count", "stage_label"]].values,
            #             hovertemplate=(
            #                 "Dimension: %{fullData.name}<br>"
            #                 "Stage: %{customdata[1]}<br>"
            #                 "Mean direction: %{y:.1f}%<br>"
            #                 "Queries: %{customdata[0]}<extra></extra>"
            #             ),
            #         )
            #     )

            per_platform_line_fig.update_layout(
                title=f"{display_name}: specificity direction by dimension",
                xaxis_title="Query Formulation Iteration",
                yaxis_title="Avg Specificity Increase (%)",
                margin=dict(t=50, b=80, l=80, r=30),
                legend_title="",
                # legend=dict(font=dict(size=20)),
            )
            per_platform_line_fig.update_xaxes(
                tickmode="array",
                tickvals=[1, 2, 3],
                ticktext=[
                    "User → Iter. 1",
                    "Iter. 1 → Iter. 2",
                    "Iter. 2+ → Iter. 3+",
                ],
                range=[0.8, 3.4],
                tickfont=dict(size=21),
            )
            per_platform_line_fig.update_yaxes(ticksuffix="%")
            per_platform_file_name = (
                f"{dimension_output_file_name}__{platform}"
            )
            per_platform_line_fig = with_paper_style(
                per_platform_line_fig,
                config=styler(24, 24),
                # legend_pos=(0.9, 1.18),
            )
            per_platform_line_fig.write_image(
                f"{OUTPUT_PATH}/{platform}/{CONF}/{per_platform_file_name}.pdf",
                format="pdf",
            )

    return plot_df


def query_specificity_evaluation(platform="chatgpt"):
    output_stem = "query_reformulations_query_specificity"
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/query_reformulation_with_thought_src_mem.pkl"
    )
    df = _filter_query_reformulation_df(df)

    specificity_dimensions = {
        "temporal": (
            SYSTEM_PROMPT_TEMPORAL_SPECIFICITY,
            USER_PROMPT_TEMPORAL_SPECIFICITY,
        ),
        "geographic": (
            SYSTEM_PROMPT_GEOGRAPHIC_SPECIFICITY,
            USER_PROMPT_GEOGRAPHIC_SPECIFICITY,
        ),
        "entity": (
            SYSTEM_PROMPT_ENTITY_SPECIFICITY,
            USER_PROMPT_ENTITY_SPECIFICITY,
        ),
        "numeric": (
            SYSTEM_PROMPT_NUMERIC_SPECIFICITY,
            USER_PROMPT_NUMERIC_SPECIFICITY,
        ),
    }

    def _evaluate_query_specificity(query_text):
        query_text = str(query_text or "").strip()
        specificity = {}
        if not query_text:
            return specificity

        for dimension, (system_prompt, user_prompt_template) in (
            specificity_dimensions.items()
        ):
            eval_result = run_judge(
                platform,
                system_prompt=system_prompt,
                user_prompt=user_prompt_template.format(QUERY=query_text),
            )
            parsed = eval_result["parsed_judgment"]
            if not isinstance(parsed, dict):
                parsed = {}

            score = parsed.get("score")
            try:
                score = int(score)
            except (TypeError, ValueError):
                score = None

            specificity[dimension] = {
                "score": score,
                "reason": str(parsed.get("reason", "")).strip(),
            }

        return specificity

    records = []
    print(f"Loaded {len(df)} rows")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_query = _row_latest_user_query(row)
        if not user_query:
            continue

        web_query_groups = _safe_json_value(row.get("web_queries"), [])

        try:
            user_query_specificity = _evaluate_query_specificity(user_query)
        except Exception as e:
            print(
                "query_specificity user_query",
                row.get("conv_id"),
                row.get("turn_id"),
                e,
            )
            continue

        web_query_specificity_info = []
        for iteration_idx, query_group in enumerate(web_query_groups, start=1):
            if isinstance(query_group, list):
                current_queries = query_group
            else:
                current_queries = [query_group]

            for web_query in current_queries:
                if not isinstance(web_query, str) or not web_query.strip():
                    continue

                web_query = web_query.strip()
                try:
                    web_query_specificity_info.append(
                        {
                            "query": web_query,
                            "iteration": iteration_idx,
                            "specificity": _evaluate_query_specificity(web_query),
                        }
                    )
                except Exception as e:
                    print(
                        "query_specificity web_query",
                        row.get("conv_id"),
                        row.get("turn_id"),
                        e,
                    )

        row_record = _base_query_record(row)
        row_record.update(
            {
                "user_query": user_query,
                "user_query_specificity": json.dumps(
                    user_query_specificity, ensure_ascii=False
                ),
                "web_queries": json.dumps(web_query_groups, ensure_ascii=False),
                "web_query_specificity_info": json.dumps(
                    web_query_specificity_info, ensure_ascii=False
                ),
            }
        )
        records.append(row_record)

        if len(records) % 20 == 0:
            _save_query_eval_records(records, output_stem, platform=platform)

    return _save_query_eval_records(
        records,
        output_stem=output_stem,
        platform=platform,
    )



if __name__ == "__main__":
    # Run every per-platform analysis once for each platform we have
    # extracted data for (see src.utils.common_io.PLATFORMS); each writes
    # under its own outputs/<platform>/query_reformulations/ so results
    # from different platforms never overwrite each other.
    for platform in PLATFORMS:
        web_df = load_web_data_from_file(fmt="pkl", platform=platform)
        print(f"[{platform}] Loaded web data: {len(web_df)}")
        if platform == "chatgpt":
            gather_query_reform_effective_factors(web_df)
        else:
            gather_query_reform_effective_factors_other_platforms(web_df, platform)

        web_query_tokens_source_detection(platform=platform)
        plot_web_query_tokens_source_detection(platform=platform)

        # fanouts and iterations
        plot_number_of_loops_histogram(platform=platform)
        plot_number_of_fanout_queries_and_iterations_over_time(platform=platform)

        query_specificity_evaluation(platform=platform)
        plot_query_specificity_distribution_by_iteration(platform=platform)

    # Cross-platform comparison plot: combines all four platforms into one
    # figure by design, so it runs once (not per platform) -- it loads all
    # 4 platforms' data itself, see its own docstring/platform_configs.
    plot_query_term_count_trends_over_time(remove_stopwords=False)
