"""§4.3/§5.1 analyses: retrieved and cited source URLs -- domain preferences
and bias, Tranco/topical authority ranking, citation-vs-retrieval rates,
reachability/hallucination checks, and how much evidence (URL counts per
query/response) each platform's search engine actually returns.

Same scope note as query_reformulations.py: written for the paper's full
cohort, organized as a library of individually-runnable analysis functions
(see the mostly-commented-out call list in `if __name__ == "__main__"`),
each writing its own figure/table under outputs/source_selection/.
"""

import os
import ast
import json
import re
from tqdm import tqdm
import pandas as pd
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import ttest_rel
import numpy as np
from collections import Counter
pio.defaults.mathjax = None
from src.utils.common_io import *
from src.utils.chatgpt_conversation_utils import *
from src.utils.figure_style import with_paper_style, styler
from src.web_search_decision.extraction import load_web_data_from_file
from src.response_generation.web_content_fetch import _load_response_source_similarity_input
import tiktoken

CONF = "./source_selection"
TIMEOUT = 5
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}


def extract_retrieved_cited_source(web_df):
    outer_pattern = r"\ue200(?=[^\ue201]*\ue202[A-Za-z]+\d+[A-Za-z]+\d+(?:\ue202|\ue201))[^\ue201]*\ue201"
    inner_pattern = r"\ue202[A-Za-z]+(\d+)[A-Za-z]+(\d+)(?=\ue202|\ue201)"

    def _dedupe_cited_items(items):
        def _item_richness(item):
            score = 0
            for value in item.values():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                score += 1
            return score

        unique_items = []
        key_to_index = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            dedupe_key = (
                item.get("url", ""),
                item.get("ref_index", None),
                item.get("turn_index", None),
            )
            existing_index = key_to_index.get(dedupe_key)
            if existing_index is None:
                key_to_index[dedupe_key] = len(unique_items)
                unique_items.append(item)
                continue

            if _item_richness(item) > _item_richness(unique_items[existing_index]):
                unique_items[existing_index] = item
        return unique_items

    web_df["srcs_retrieved"] = [{}] * len(web_df)
    web_df["srcs_cited"] = [{}] * len(web_df)

    for i, row in tqdm(web_df.iterrows()):
        msgs = json.loads(row['turn_msgs'])
        srcs_retrieved = []
        srcs_cited = []
        for msg in msgs:
            # retrieved
            retrieved = msg.get('metadata', {}).get('search_result_groups', [])
            for r in retrieved:
                entries = r.get("entries", [])
                for entry in entries:
                    url = entry.get("url", "")
                    if url:
                        d = urlparse(entry['url']).netloc.replace("www.", "")
                        srcs_retrieved.append(
                            {
                                "url": url,
                                "domain": d,
                                "title": entry.get("title", ""),
                                "ref_index": entry.get("ref_id", {}).get("ref_index", None) if entry.get("ref_id", {}) else None,
                                "turn_index": entry.get("ref_id", {}).get("turn_index", None) if entry.get("ref_id", {}) else None,
                                "snippet": entry["snippet"],
                            }
                        )

            retrieved = msg.get('metadata', {}).get('image_results', [])
            for ri, r in enumerate(retrieved):
                d = urlparse(r["url"]).netloc.replace("www.", "")
                srcs_retrieved.append(
                    {
                        "url": r["url"],
                        "domain": d,
                        "title": r.get("title", ""),
                        "ref_index": ri,
                    }
                )

            # cited
            cited = msg.get('metadata', {}).get('content_references', [])
            for r in cited:
                matched_text = r.get("matched_text", "").strip()
                if matched_text:
                    outer = re.search(outer_pattern, matched_text)
                    if outer:
                        found_refs = re.findall(inner_pattern, outer.group(0))
                        cited_turns = []
                        cited_ranks = []
                        for fr in found_refs:
                            cited_turns.append(int(fr[0]))
                            cited_ranks.append(int(fr[1]))

                        # print(matched_text, cited_turns, cited_ranks)

                        url = r.get("url", "")
                        if url:
                            url = url.removesuffix("?utm_source=chatgpt.com").removesuffix("&utm_source=chatgpt.com")
                            d = urlparse(url).netloc.replace("www.", "")
                            srcs_cited.append(
                                {
                                    "url": url,
                                    "domain": d,
                                    "title": r.get("title"),
                                    "snippet": r.get("snippet"),
                                    "ref_index": cited_ranks[0],
                                    "turn_index": cited_turns[0]
                                }
                            )

                            
                        if "fallback_items" in r and r["fallback_items"]:
                            keys_to_check = ["images", "fallback_items"]
                        else:
                            keys_to_check = ["images", "items"]

                        for key in keys_to_check:
                            items = r.get(key, [])
                            refs = r.get("refs", [])
                            if items:
                                for ii, item in enumerate(items):
                                    url = item.get("url", "").removesuffix("?utm_source=chatgpt.com").removesuffix("&utm_source=chatgpt.com")
                                    d = urlparse(url).netloc.replace("www.", "")
                                    if item.get("refs", []):
                                        ref = item.get("refs", [])[0]
                                    else:
                                        ref = refs[ii] if ii < len(refs) else {}
                                    if url:
                                        srcs_cited.append(
                                            {
                                                "url": url,
                                                "domain": d,
                                                "title": item.get("title", ""),
                                                "snippet": item.get("snippet", ""),
                                                "ref_index": ref.get("ref_index", None),
                                                "turn_index": ref.get("turn_index", None)
                                            }
                                        )


        web_df.at[i, "srcs_retrieved"] = srcs_retrieved
        web_df.at[i, "srcs_cited"] = _dedupe_cited_items(srcs_cited)

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)

    web_df.to_csv(
        f"{OUTPUT_PATH}/chatgpt/metadata/retrieved_cited_extracted_from_srcs.csv",
        index=False,
    )
    web_df.to_pickle(
        f"{OUTPUT_PATH}/chatgpt/metadata/retrieved_cited_extracted_from_srcs.pkl"
    )

def _unique_source_count(items, key="url"):
    if not isinstance(items, list):
        return 0
    values = {
        item.get(key, "")
        for item in items
        if isinstance(item, dict) and item.get(key, "")
    }
    return len(values)


def _primary_model(models):
    if not isinstance(models, list):
        return "Unknown"
    cleaned = [model for model in models if isinstance(model, str) and model]
    if not cleaned:
        return "Unknown"
    return cleaned[-1]


def _prepare_source_count_df(model=""):
    """Load outputs/<model>/metadata/response_and_sources.pkl (model=""
    defaults to "chatgpt") and add per-row retrieved/cited URL counts.

    Pipeline dependency: this file is NOT produced by anything in this
    module. Run response_generation.extract_response_and_sources(web_df)
    (or extract_response_and_sources_other_platforms() for non-ChatGPT
    platforms) first -- it's a separate scrape/dedupe pass over the raw
    turn messages, distinct from extract_retrieved_cited_source() above,
    which writes a differently-named file
    (retrieved_cited_extracted_from_srcs.pkl) that functions in *this*
    module read instead. The two pipelines are not interchangeable inputs
    for each other despite the similar-sounding names.
    """
    if model == "chatgpt" or model == "":
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/chatgpt/metadata/response_and_sources.pkl"
        ).copy()
    else:
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/{model}/metadata/response_and_sources.pkl"
        ).copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    if "models" in df.columns:
        df["model"] = df["models"].apply(_primary_model)
    df["num_retrieved_urls"] = df["srcs_retrieved"].apply(_unique_source_count)
    df["num_cited_urls"] = df["srcs_cited"].apply(_unique_source_count)
    return df


def count_unique_retrieved_cited(platform="chatgpt"):
    """Count of responses with at least one retrieved/cited URL each.
    Requires response_generation.extract_response_and_sources[_other_platforms]()
    to have been run first -- see _prepare_source_count_df()'s docstring."""
    df = _prepare_source_count_df(platform)
    return {
        "retrieved_urls": (df["num_retrieved_urls"] > 0).sum(),
        "cited_urls": (df["num_cited_urls"] > 0).sum(),
    }


def _source_item_count(items, key="url", unique=True):
    if unique:
        return _unique_source_count(items, key=key)
    if not isinstance(items, list):
        return 0
    return sum(
        1
        for item in items
        if isinstance(item, dict) and (key is None or item.get(key, ""))
    )


def _bootstrap_mean_ci(values, confidence=0.95, n_bootstrap=10000, random_seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    mean = values.mean()
    if n == 1:
        return {
            "n": 1,
            "mean": float(mean),
            "ci_low": float(mean),
            "ci_high": float(mean),
        }

    rng = np.random.default_rng(random_seed)
    bootstrap_indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    alpha = 1.0 - confidence
    ci_low = np.quantile(bootstrap_means, alpha / 2.0)
    ci_high = np.quantile(bootstrap_means, 1.0 - alpha / 2.0)

    return {
        "n": int(n),
        "mean": float(mean),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _build_query_url_counts_from_response(
    web_query_groups,
    retrieved_source_items,
    unique=False,
):
    total_queries = _web_query_groups_to_count(web_query_groups)
    if total_queries <= 0:
        return []

    retrieved_count = _source_item_count(
        retrieved_source_items,
        key="url",
        unique=unique,
    )
    per_query_count = retrieved_count / total_queries
    return [per_query_count] * int(total_queries)


def _bootstrap_ratio_ci_by_conversation(
    conv_ids,
    numerator_values,
    denominator_values,
    confidence=0.95,
    n_bootstrap=10000,
    random_seed=0,
):
    conv_ids = np.asarray(conv_ids, dtype=object)
    numerator_values = np.asarray(numerator_values, dtype=float)
    denominator_values = np.asarray(denominator_values, dtype=float)

    valid_mask = (
        pd.notna(conv_ids)
        & np.isfinite(numerator_values)
        & np.isfinite(denominator_values)
    )
    conv_ids = conv_ids[valid_mask]
    numerator_values = numerator_values[valid_mask]
    denominator_values = denominator_values[valid_mask]

    if len(conv_ids) == 0:
        return {
            "n_conversations": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    conv_df = pd.DataFrame(
        {
            "conv_id": conv_ids,
            "numerator": numerator_values,
            "denominator": denominator_values,
        }
    )
    conv_agg = (
        conv_df.groupby("conv_id", dropna=False)[["numerator", "denominator"]]
        .sum()
        .reset_index()
    )
    if len(conv_agg) == 0:
        return {
            "n_conversations": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    total_denominator = float(conv_agg["denominator"].sum())
    point_estimate = (
        float(conv_agg["numerator"].sum() / total_denominator)
        if total_denominator > 0
        else np.nan
    )

    n_conv = len(conv_agg)
    rng = np.random.default_rng(random_seed)
    bootstrap_indices = rng.integers(0, n_conv, size=(n_bootstrap, n_conv))
    numerators = conv_agg["numerator"].to_numpy()[bootstrap_indices].sum(axis=1)
    denominators = conv_agg["denominator"].to_numpy()[bootstrap_indices].sum(axis=1)
    bootstrap_ratios = np.divide(
        numerators,
        denominators,
        out=np.full(n_bootstrap, np.nan, dtype=float),
        where=denominators > 0,
    )
    bootstrap_ratios = bootstrap_ratios[np.isfinite(bootstrap_ratios)]
    if len(bootstrap_ratios) == 0:
        ci_low = np.nan
        ci_high = np.nan
    else:
        alpha = 1.0 - confidence
        ci_low = float(np.quantile(bootstrap_ratios, alpha / 2.0))
        ci_high = float(np.quantile(bootstrap_ratios, 1.0 - alpha / 2.0))

    return {
        "n_conversations": int(n_conv),
        "mean": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _web_query_groups_to_count(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = []
    if not isinstance(value, list):
        return 0

    total = 0
    for query_group in value:
        if isinstance(query_group, list):
            total += sum(
                1 for query in query_group
                if isinstance(query, str) and query.strip()
            )
        elif isinstance(query_group, str) and query_group.strip():
            total += 1
    return total


def _load_query_count_lookup(model=""):
    if model == "chatgpt" or model == "":
        candidate_paths = [
            f"{OUTPUT_PATH}/chatgpt/metadata/query_reformulation_with_thought_src_mem.pkl",
            f"{OUTPUT_PATH}/chatgpt/metadata/query_reformulation_with_thought_src_mem.csv",
        ]
    else:
        candidate_paths = [
            f"{OUTPUT_PATH}/{model}/metadata/query_reformulation_with_thought_src_mem.pkl",
            f"{OUTPUT_PATH}/{model}/metadata/query_reformulation_with_thought_src_mem.csv",
        ]

    query_df = None
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        if path.endswith(".pkl"):
            query_df = pd.read_pickle(path).copy()
        else:
            query_df = pd.read_csv(path).copy()
        break

    if query_df is None:
        return {}

    required_cols = {"user_id", "conv_id", "turn_id", "web_queries"}
    missing_cols = required_cols - set(query_df.columns)
    if missing_cols:
        return {}

    lookup = {}
    for _, row in query_df.iterrows():
        key = (
            str(row.get("user_id", "")),
            str(row.get("conv_id", "")),
            str(row.get("turn_id", "")),
        )
        lookup[key] = _web_query_groups_to_count(row.get("web_queries", []))
    return lookup


def _load_query_groups_lookup(model=""):
    if model == "chatgpt" or model == "":
        candidate_paths = [
            f"{OUTPUT_PATH}/chatgpt/metadata/query_reformulation_with_thought_src_mem.pkl",
            f"{OUTPUT_PATH}/chatgpt/metadata/query_reformulation_with_thought_src_mem.csv",
        ]
    else:
        candidate_paths = [
            f"{OUTPUT_PATH}/{model}/metadata/query_reformulation_with_thought_src_mem.pkl",
            f"{OUTPUT_PATH}/{model}/metadata/query_reformulation_with_thought_src_mem.csv",
        ]

    query_df = None
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        if path.endswith(".pkl"):
            query_df = pd.read_pickle(path).copy()
        else:
            query_df = pd.read_csv(path).copy()
        break

    if query_df is None:
        return {}

    required_cols = {"user_id", "conv_id", "turn_id", "web_queries"}
    missing_cols = required_cols - set(query_df.columns)
    if missing_cols:
        return {}

    lookup = {}
    for _, row in query_df.iterrows():
        key = (
            str(row.get("user_id", "")),
            str(row.get("conv_id", "")),
            str(row.get("turn_id", "")),
        )
        value = row.get("web_queries", [])
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    value = []
        lookup[key] = value if isinstance(value, list) else []
    return lookup


def _print_source_count_summaries_for_df(
    df,
    unique=False,
    grounding_level="turn",
    label_prefix=None,
    platform="chatgpt",
):
    query_count_lookup = _load_query_count_lookup(platform)
    query_groups_lookup = _load_query_groups_lookup(platform)
    query_keys = set(query_count_lookup.keys())
    df = df.copy()
    df["web_queries"] = df.apply(
        lambda row: query_groups_lookup.get(
            (
                str(row.get("user_id", "")),
                str(row.get("conv_id", "")),
                str(row.get("turn_id", "")),
            ),
            [],
        ),
        axis=1,
    )

    retrieved_counts = df["srcs_retrieved"].apply(
        lambda items: _source_item_count(items, key="url", unique=unique)
    )
    cited_counts = df["srcs_cited"].apply(
        lambda items: _source_item_count(items, key="url", unique=unique)
    )
    query_trace_counts = df.apply(
        lambda row: query_count_lookup.get(
            (
                str(row.get("user_id", "")),
                str(row.get("conv_id", "")),
                str(row.get("turn_id", "")),
            ),
            np.nan,
        ),
        axis=1,
    )
    has_web_query_data = df.apply(
        lambda row: (
            str(row.get("user_id", "")),
            str(row.get("conv_id", "")),
            str(row.get("turn_id", "")),
        ) in query_keys,
        axis=1,
    )

    cited_external_counts = pd.Series(index=df.index, dtype=float)
    cited_internal_counts = pd.Series(index=df.index, dtype=float)
    previous_retrieved_by_conv = {}
    if grounding_level == "conversation":
        sort_cols = [col for col in ["conv_id", "turn_id", "time"] if col in df.columns]
        grounding_df = df.sort_values(sort_cols, kind="stable") if sort_cols else df
    else:
        grounding_df = df

    for row_idx, row in grounding_df.iterrows():
        retrieved_sources = row.get("srcs_retrieved", [])
        cited_sources = row.get("srcs_cited", [])
        current_retrieved_urls = _normalized_urls_from_sources(retrieved_sources)
        cited_urls = _normalized_urls_from_sources(cited_sources)

        if unique:
            current_retrieved_urls = set(current_retrieved_urls)
            cited_urls = list(dict.fromkeys(cited_urls))
        else:
            current_retrieved_urls = set(current_retrieved_urls)

        if grounding_level == "conversation":
            conv_id = row.get("conv_id", None)
            previous_retrieved_urls = previous_retrieved_by_conv.get(conv_id, set())
            retrieved_urls = previous_retrieved_urls | current_retrieved_urls
            previous_retrieved_by_conv[conv_id] = retrieved_urls
        else:
            retrieved_urls = current_retrieved_urls

        cited_external_count = sum(1 for url in cited_urls if url in retrieved_urls)
        cited_external_counts.at[row_idx] = cited_external_count
        cited_internal_counts.at[row_idx] = len(cited_urls) - cited_external_count

    sum_retrievals = float(retrieved_counts.sum())
    sum_citations = float(cited_counts.sum())
    sum_cited_external = int(cited_external_counts.fillna(0).sum())
    sum_cited_internal = int(cited_internal_counts.fillna(0).sum())
    all_samples_with_web_call = np.ones(len(df), dtype=bool)

    def _print_summary(summary_name, sample_mask, include_query_metrics):
        mask_array = np.asarray(sample_mask, dtype=bool)
        sample_conv_ids = df["conv_id"][mask_array].to_numpy()
        sample_retrieved_counts = retrieved_counts[mask_array]
        sample_cited_counts = cited_counts[mask_array]
        sample_query_trace_counts = query_trace_counts[mask_array]
        sample_cited_external_counts = cited_external_counts[mask_array].to_numpy()
        sample_cited_internal_counts = cited_internal_counts[mask_array].to_numpy()
        sum_sample_retrievals = float(sample_retrieved_counts.sum())
        sum_sample_citations = float(sample_cited_counts.sum())
        sum_sample_cited_external = int(np.sum(sample_cited_external_counts))
        sum_sample_cited_internal = int(np.sum(sample_cited_internal_counts))

        summary = {
            "summary": summary_name,
            "num_responses": int(sample_mask.sum()),
            "sum_retrievals": sum_sample_retrievals,
            "avg_retrievals_per_response": (
                float(sample_retrieved_counts.mean())
                if len(sample_retrieved_counts) > 0
                else 0.0
            ),
            "sum_citations": sum_sample_citations,
            "avg_citations_per_response": (
                float(sample_cited_counts.mean())
                if len(sample_cited_counts) > 0
                else 0.0
            ),
            "grounding_level": grounding_level,
            "sum_cited_external": sum_sample_cited_external,
            "sum_cited_internal": sum_sample_cited_internal,
            "citation_rate": (
                sum_sample_cited_external / sum_sample_retrievals
                if sum_sample_retrievals > 0
                else 0.0
            ),
            "grounding_rate": (
                sum_sample_cited_external / sum_sample_citations
                if sum_sample_citations > 0
                else 0.0
            ),
        }

        if include_query_metrics:
            sum_queries_issued = int(sample_query_trace_counts.fillna(0).sum())
            query_url_counts = []
            for _, row in df[sample_mask].iterrows():
                query_url_counts.extend(
                    _build_query_url_counts_from_response(
                        row.get("web_queries", []),
                        row.get("srcs_retrieved", []),
                        unique=unique,
                    )
                )
            summary.update({
                "sum_queries_issued": sum_queries_issued,
                "avg_queries_per_response": (
                    float(sample_query_trace_counts.mean())
                    if len(sample_query_trace_counts) > 0
                    else 0.0
                ),
                "avg_retrieved_urls_per_query": (
                    float(np.mean(query_url_counts))
                    if len(query_url_counts) > 0
                    else 0.0
                ),
                "sum_retrievals_in_query_sample": float(np.sum(query_url_counts)),
                "avg_web_queries_per_user_query_ci_95": _bootstrap_mean_ci(
                    sample_query_trace_counts
                ),
                "avg_urls_per_user_query_ci_95": _bootstrap_mean_ci(
                    sample_retrieved_counts
                ),
                "avg_urls_per_web_query_ci_95": _bootstrap_mean_ci(
                    query_url_counts
                ),
            })

        summary["citation_rate_ci_95"] = _bootstrap_ratio_ci_by_conversation(
            sample_conv_ids,
            sample_cited_external_counts,
            sample_retrieved_counts.to_numpy(),
        )
        summary["grounding_rate_ci_95"] = _bootstrap_ratio_ci_by_conversation(
            sample_conv_ids,
            sample_cited_external_counts,
            sample_cited_counts.to_numpy(),
        )

        if label_prefix:
            print(label_prefix)
        print(summary)

    _print_summary(
        "all_samples_with_web_call",
        all_samples_with_web_call,
        include_query_metrics=False,
    )
    _print_summary(
        "samples_with_web_queries",
        has_web_query_data,
        include_query_metrics=True,
    )


def compute_average_citations_and_retrievals_per_response(
    unique=False,
    grounding_level="turn",
):
    """Compute average cited and retrieved sources per response.

    By default this counts raw source records. Set unique=True to count unique
    URLs instead.

    grounding_level controls how cited URLs are classified:
    - "turn": cited URL must appear in the same turn's retrieved URLs.
    - "conversation": cited URL may appear in the current turn or any earlier
      retrieved turn from the same conversation.
    """
    grounding_level = _validated_grounding_level(grounding_level)

    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)

    for model in PLATFORMS:
        print(model)
        df = _prepare_source_count_df(model)
        _print_source_count_summaries_for_df(
            df,
            unique=unique,
            grounding_level=grounding_level,
            platform=model,
        )


def compute_average_citations_and_retrievals_per_response_by_model(
    unique=False,
    grounding_level="turn",
):
    grounding_level = _validated_grounding_level(grounding_level)
    df = _prepare_source_count_df("chatgpt").copy()
    if "models" not in df.columns:
        print("No `models` column in response_and_sources.pkl; skipping.")
        return
    df["model"] = df["models"].apply(_primary_model)
    df = df[df["model"].str.lower() != "unknown"].copy()

    for model_name in sorted(df["model"].dropna().unique()):
        model_df = df[df["model"] == model_name].copy()
        if len(model_df) == 0:
            continue
        _print_source_count_summaries_for_df(
            model_df,
            unique=unique,
            grounding_level=grounding_level,
            label_prefix=f"chatgpt::{model_name}",
        )


def plot_retrieved_urls_per_web_query_histogram(unique=False, bins=40):
    for model in PLATFORMS:
        df = _prepare_source_count_df(model)
        df = df.copy()
        query_count_lookup = _load_query_count_lookup(model)
        query_groups_lookup = _load_query_groups_lookup(model)
        query_keys = set(query_count_lookup.keys())
        df["web_queries"] = df.apply(
            lambda row: query_groups_lookup.get(
                (
                    str(row.get("user_id", "")),
                    str(row.get("conv_id", "")),
                    str(row.get("turn_id", "")),
                ),
                [],
            ),
            axis=1,
        )

        retrieved_counts = df["srcs_retrieved"].apply(
            lambda items: _source_item_count(items, key="url", unique=unique)
        )
        query_trace_counts = df.apply(
            lambda row: query_count_lookup.get(
                (
                    str(row.get("user_id", "")),
                    str(row.get("conv_id", "")),
                    str(row.get("turn_id", "")),
                ),
                np.nan,
            ),
            axis=1,
        )
        has_web_query_data = df.apply(
            lambda row: (
                str(row.get("user_id", "")),
                str(row.get("conv_id", "")),
                str(row.get("turn_id", "")),
            ) in query_keys,
            axis=1,
        )

        query_url_counts = []
        for _, row in df[has_web_query_data].iterrows():
            query_url_counts.extend(
                _build_query_url_counts_from_response(
                    row.get("web_queries", []),
                    row.get("srcs_retrieved", []),
                    unique=unique,
                )
            )
        if not query_url_counts:
            continue

        print(model)
        print(len(query_url_counts))
        print(float(np.mean(query_url_counts)))
        print(float(np.sum(query_url_counts)))

        plot_df = pd.DataFrame(
            {"retrieved_urls_per_web_query": query_url_counts}
        )
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=plot_df["retrieved_urls_per_web_query"],
                nbinsx=bins,
                marker_color="#3765E5",
                showlegend=False,
            )
        )
        fig.update_layout(
            xaxis_title="Retrieved URLs per Web Query",
            yaxis_title="Count",
            bargap=0.08,
            margin=dict(l=70, b=70, t=40, r=30),
        )

        model_output_dir = f"{OUTPUT_PATH}/{model}/{CONF}"
        os.makedirs(model_output_dir, exist_ok=True)
        file_name = "retrieved_urls_per_web_query_histogram"
        pdf_path = f"{model_output_dir}/{file_name}.pdf"
        fig = with_paper_style(fig, config=styler(18, 16), legend_pos=None)
        try:
            print(f"Writing histogram PDF: {pdf_path}")
            fig.write_image(pdf_path, format="pdf")
        except Exception as exc:
            print(f"Failed to write histogram PDF for {model}: {exc}")

def _normalize_domain_for_top_plots(domain):
    if not isinstance(domain, str):
        return ""
    normalized = domain.strip().lower().rstrip(".")
    if normalized.startswith("www."):
        normalized = normalized[4:]
    if normalized == "wikipedia.org" or normalized.endswith(".wikipedia.org"):
        return "wikipedia.org"
    return normalized


def _normalize_url_for_source_matching(url):
    if not isinstance(url, str):
        return ""
    normalized = (
        url.strip()
        .removesuffix("?utm_source=chatgpt.com")
        .removesuffix("&utm_source=chatgpt.com")
        .removesuffix("?utm_source=openai")
        .removesuffix("&utm_source=openai")
        .rstrip("/")
    )
    return normalized


def _validated_grounding_level(grounding_level):
    valid_grounding_levels = {"turn", "conversation"}
    if grounding_level not in valid_grounding_levels:
        raise ValueError(
            f"grounding_level must be one of {sorted(valid_grounding_levels)}, "
            f"got {grounding_level!r}"
        )
    return grounding_level


def _normalized_urls_from_sources(sources):
    if not isinstance(sources, list):
        return []
    return [
        _normalize_url_for_source_matching(src.get("url", ""))
        for src in sources
        if isinstance(src, dict) and src.get("url")
    ]


def _iter_rows_for_grounding(df, grounding_level):
    grounding_level = _validated_grounding_level(grounding_level)
    if grounding_level == "turn":
        return df.itertuples(index=False)

    sort_cols = [col for col in ["conv_id", "turn_id", "time"] if col in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols, kind="stable").itertuples(index=False)
    return df.itertuples(index=False)


def _row_retrieved_urls_by_grounding(row, previous_retrieved_by_conv, grounding_level):
    current_retrieved_urls = set(
        _normalized_urls_from_sources(getattr(row, "srcs_retrieved", []))
    )
    if grounding_level == "conversation":
        conv_id = getattr(row, "conv_id", None)
        previous_retrieved_urls = previous_retrieved_by_conv.get(conv_id, set())
        retrieved_urls = previous_retrieved_urls | current_retrieved_urls
        previous_retrieved_by_conv[conv_id] = retrieved_urls
        return retrieved_urls
    return current_retrieved_urls


def _domain_counter(df, col_name, top_k=20):

    counts = {}
    for items in df[col_name]:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            domain = _normalize_domain_for_top_plots(item.get("domain", ""))
            if domain:
                counts[domain] = counts.get(domain, 0) + 1

    plot_df = pd.DataFrame(
        {"domain": list(counts.keys()), "count": list(counts.values())}
    )
    if len(plot_df) == 0:
        return plot_df
    plot_df = plot_df.sort_values("count", ascending=False)
    if top_k is None:
        return plot_df
    return plot_df.head(top_k)


def _cited_domain_counter_split(df, top_k=20, grounding_level="turn"):
    grounding_level = _validated_grounding_level(grounding_level)
    external_counts = {}
    internal_counts = {}
    previous_retrieved_by_conv = {}

    for row in _iter_rows_for_grounding(df, grounding_level):
        retrieved_items = getattr(row, "srcs_retrieved", [])
        cited_items = getattr(row, "srcs_cited", [])
        if not isinstance(retrieved_items, list) or not isinstance(cited_items, list):
            continue

        retrieved_urls = _row_retrieved_urls_by_grounding(
            row,
            previous_retrieved_by_conv,
            grounding_level,
        )

        for item in cited_items:
            if not isinstance(item, dict):
                continue
            cited_url = _normalize_url_for_source_matching(item.get("url", ""))
            if not cited_url:
                # Cannot classify as external/internal without a cited URL.
                continue

            domain = _normalize_domain_for_top_plots(item.get("domain", ""))
            if not domain:
                domain = _normalize_domain_for_top_plots(urlparse(cited_url).netloc)
            if not domain:
                continue

            # External/internal split must be determined via URL overlap only.
            is_external = cited_url in retrieved_urls

            if is_external:
                external_counts[domain] = external_counts.get(domain, 0) + 1
            else:
                internal_counts[domain] = internal_counts.get(domain, 0) + 1

    domains = sorted(set(external_counts.keys()) | set(internal_counts.keys()))
    if not domains:
        return pd.DataFrame(
            columns=["domain", "external_count", "internal_count", "total_count"]
        )

    rows = []
    for domain in domains:
        external = int(external_counts.get(domain, 0))
        internal = int(internal_counts.get(domain, 0))
        rows.append(
            {
                "domain": domain,
                "external_count": external,
                "internal_count": internal,
                "total_count": external + internal,
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("total_count", ascending=False)
    if top_k is None:
        return plot_df
    return plot_df.head(top_k)


def _load_domain_plot_df(platform=""):
    if platform and platform != "chatgpt":
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources.pkl"
        ).copy()
    else:
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/chatgpt/metadata/retrieved_cited_extracted_from_srcs.pkl"
        ).copy()
    df = df[
        df["srcs_retrieved"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        & df["srcs_cited"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()
    return df


def _write_figure(
    fig,
    output_dir,
    file_name,
    paper_config=styler(18, 12),
    legend_pos=(0.8, 1.2),
    new_legend=None,
    x_tickfont_size=None,
    y_tickfont_size=10,
):
    os.makedirs(output_dir, exist_ok=True)
    fig = with_paper_style(
        fig,
        config=paper_config,
        legend_pos=legend_pos,
        new_legend=new_legend,
    )
    if x_tickfont_size is None:
        if "20" in file_name:
            x_tickfont_size = 12
        else:
            x_tickfont_size = 12
    fig.update_xaxes(tickfont=dict(size=x_tickfont_size))
    fig.update_yaxes(tickfont=dict(size=y_tickfont_size))
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

def plot_top_domains(
    separate_cited_external_internal=False,
    grounding_level="turn",
    top_k=10,
    platform="chatgpt",
):
    grounding_level = _validated_grounding_level(grounding_level)
    df = _load_domain_plot_df(platform=platform)
    output_dir = f"{OUTPUT_PATH}/{platform}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)

    if separate_cited_external_internal:
        subplot_titles = [
            "Top Search Results Domains",
            "Top Cited Search Results Domains",
            "Top Cited Parametric Domains",
        ]
    else:
        subplot_titles = [
            "Top Search Results Domains",
            "Top Cited Domains",
        ]

    fig = make_subplots(
        rows=len(subplot_titles),
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.30,
    )
    fig.update_annotations(font_size=22)

    retrieved_all_df = _domain_counter(df, "srcs_retrieved", top_k=None)
    retrieved_all_df.to_csv(
        f"{output_dir}/retrieved_domains_sorted.csv",
        index=False,
    )
    retrieved_denominator = float(retrieved_all_df["count"].sum()) if len(retrieved_all_df) > 0 else 0.0
    retrieved_df = retrieved_all_df.head(top_k)
    if retrieved_denominator > 0 and len(retrieved_df) > 0:
        fig.add_trace(
            go.Bar(
                x=retrieved_df["domain"],
                y=retrieved_df["count"] / retrieved_denominator,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if separate_cited_external_internal:
        split_df = _cited_domain_counter_split(
            df,
            top_k=None,
            grounding_level=grounding_level,
        )
        split_df.to_csv(
            (
                f"{output_dir}/"
                f"cited_domains_split_{grounding_level}_grounding_sorted.csv"
            ),
            index=False,
        )
        split_df[split_df["external_count"] > 0][["domain", "external_count"]].sort_values(
            "external_count",
            ascending=False,
        ).to_csv(
            (
                f"{output_dir}/"
                f"cited_external_domains_{grounding_level}_grounding_sorted.csv"
            ),
            index=False,
        )
        split_df[split_df["internal_count"] > 0][["domain", "internal_count"]].sort_values(
            "internal_count",
            ascending=False,
        ).to_csv(
            (
                f"{output_dir}/"
                f"cited_internal_domains_{grounding_level}_grounding_sorted.csv"
            ),
            index=False,
        )
        cited_external_denominator = (
            float(split_df["external_count"].sum()) if len(split_df) > 0 else 0.0
        )
        cited_internal_denominator = (
            float(split_df["internal_count"].sum()) if len(split_df) > 0 else 0.0
        )

        external_df = split_df[split_df["external_count"] > 0].copy()
        external_df = external_df.sort_values("external_count", ascending=False).head(top_k)
        if cited_external_denominator > 0 and len(external_df) > 0:
            fig.add_trace(
                go.Bar(
                    x=external_df["domain"],
                    y=external_df["external_count"] / cited_external_denominator,
                    marker_color="#00CC96",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        internal_df = split_df[split_df["internal_count"] > 0].copy()
        internal_df = internal_df.sort_values("internal_count", ascending=False).head(top_k)
        if cited_internal_denominator > 0 and len(internal_df) > 0:
            fig.add_trace(
                go.Bar(
                    x=internal_df["domain"],
                    y=internal_df["internal_count"] / cited_internal_denominator,
                    marker_color="#E45756",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
    else:
        cited_all_df = _domain_counter(df, "srcs_cited", top_k=None)
        cited_all_df.to_csv(
            f"{output_dir}/cited_domains_sorted.csv",
            index=False,
        )
        cited_denominator = float(cited_all_df["count"].sum()) if len(cited_all_df) > 0 else 0.0
        cited_df = cited_all_df.head(top_k)
        if cited_denominator > 0 and len(cited_df) > 0:
            fig.add_trace(
                go.Bar(
                    x=cited_df["domain"],
                    y=cited_df["count"] / cited_denominator,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        # height=1000,
        margin=dict(l=110, b=60, t=30, r=40),
    )
    fig.update_xaxes(
        tickangle=-20,
        automargin=True,
    )
    fig.add_annotation(
        x=-0.2,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Percentage of URLs",
        textangle=-90,
        showarrow=False,
        font=dict(size=26, color="black"),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = (
        f"top_domains_overall_split_cited_{grounding_level}_grounding"
        if separate_cited_external_internal
        else "top_domains_overall"
    )
    _write_figure(
        fig,
        output_dir,
        file_name,
        styler(18, 18),
        legend_pos=None,
        x_tickfont_size=16,
        y_tickfont_size=14,
    )


def evaluate_unique_retrieved_domains_by_platform():
    platforms = PLATFORMS
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)

    platform_dfs = {}
    platform_sets = {}
    for platform in platforms:
        path = f"{OUTPUT_PATH}/{platform}/{CONF}/retrieved_domains_sorted.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing retrieved-domain CSV for platform {platform!r}: {path}"
            )

        df = pd.read_csv(path).copy()
        required_cols = {"domain", "count"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {path}: {sorted(missing_cols)}"
            )

        df = df.dropna(subset=["domain"]).copy()
        df["domain"] = df["domain"].astype(str).map(_normalize_domain_for_top_plots)
        df = df[df["domain"] != ""].copy()
        df = df.sort_values("count", ascending=False).reset_index(drop=True)

        platform_dfs[platform] = df
        platform_sets[platform] = set(df["domain"])

    result_rows = []
    top_20_result_rows = []

    for platform in platforms:
        source_df = platform_dfs[platform]
        source_top_20_df = source_df.head(20).copy()

        other_domains_union = set()
        for other_platform in platforms:
            if other_platform == platform:
                continue
            other_domains_union |= platform_sets[other_platform]

        unique_vs_all_df = source_df[
            ~source_df["domain"].isin(other_domains_union)
        ].copy()
        for _, row in unique_vs_all_df.iterrows():
            result_rows.append(
                {
                    "platform": platform,
                    "comparison_type": "one_vs_rest",
                    "comparison_target": "all_other_platforms",
                    "domain": row["domain"],
                }
            )
        unique_top_20_vs_all_df = source_top_20_df[
            ~source_top_20_df["domain"].isin(other_domains_union)
        ].copy()
        for _, row in unique_top_20_vs_all_df.iterrows():
            top_20_result_rows.append(
                {
                    "platform": platform,
                    "comparison_type": "one_vs_rest",
                    "comparison_target": "all_other_platforms",
                    "domain": row["domain"],
                }
            )

        for other_platform in platforms:
            if other_platform == platform:
                continue

            other_domains = platform_sets[other_platform]
            unique_vs_other_df = source_df[
                ~source_df["domain"].isin(other_domains)
            ].copy()
            for _, row in unique_vs_other_df.iterrows():
                result_rows.append(
                    {
                        "platform": platform,
                        "comparison_type": "pairwise",
                        "comparison_target": other_platform,
                        "domain": row["domain"],
                    }
                )
            unique_top_20_vs_other_df = source_top_20_df[
                ~source_top_20_df["domain"].isin(other_domains)
            ].copy()
            for _, row in unique_top_20_vs_other_df.iterrows():
                top_20_result_rows.append(
                    {
                        "platform": platform,
                        "comparison_type": "pairwise",
                        "comparison_target": other_platform,
                        "domain": row["domain"],
                    }
                )

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(
        f"{output_dir}/retrieved_domains_unique_platform_comparisons.csv",
        index=False,
    )
    top_20_result_df = pd.DataFrame(top_20_result_rows)
    top_20_result_df.to_csv(
        f"{output_dir}/retrieved_domains_unique_platform_comparisons_top_20.csv",
        index=False,
    )
    return result_df


def plot_url_counts_over_time(separate_cited_external_internal=False, platform="chatgpt"):
    df = _prepare_source_count_df(platform)

    if separate_cited_external_internal:
        def _count_cited_external_internal_urls(row):
            retrieved_urls = {
                _normalize_url_for_source_matching(item.get("url", ""))
                for item in row.get("srcs_retrieved", [])
                if isinstance(item, dict) and item.get("url", "")
            }
            cited_external_urls = set()
            cited_internal_urls = set()
            for item in row.get("srcs_cited", []):
                if not isinstance(item, dict):
                    continue
                cited_url = _normalize_url_for_source_matching(item.get("url", ""))
                if not cited_url:
                    continue
                if cited_url in retrieved_urls:
                    cited_external_urls.add(cited_url)
                else:
                    cited_internal_urls.add(cited_url)
            return pd.Series(
                {
                    "num_cited_external_urls": len(cited_external_urls),
                    "num_cited_internal_urls": len(cited_internal_urls),
                }
            )

        df = df.copy()
        df[["num_cited_external_urls", "num_cited_internal_urls"]] = df.apply(
            _count_cited_external_internal_urls,
            axis=1,
        )
        value_cols = [
            "num_retrieved_urls",
            "num_cited_external_urls",
            "num_cited_internal_urls",
        ]
    else:
        value_cols = ["num_retrieved_urls", "num_cited_urls"]

    monthly = (
        df.groupby("month")[value_cols]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly[("num_retrieved_urls", "mean")],
            mode="lines+markers",
            name="Retrieved URLs",
            error_y=dict(
                type="data",
                array=monthly[("num_retrieved_urls", "sem")].fillna(0),
                visible=True,
            ),
        )
    )
    if separate_cited_external_internal:
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly[("num_cited_internal_urls", "mean")],
                mode="lines+markers",
                name="Cited Unexplained/Internal URLs",
                error_y=dict(
                    type="data",
                    array=monthly[("num_cited_internal_urls", "sem")].fillna(0),
                    visible=True,
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly[("num_cited_external_urls", "mean")],
                mode="lines+markers",
                name="Cited Retrieved/External URLs",
                error_y=dict(
                    type="data",
                    array=monthly[("num_cited_external_urls", "sem")].fillna(0),
                    visible=True,
                ),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly[("num_cited_urls", "mean")],
                mode="lines+markers",
                name="Cited URLs",
                error_y=dict(
                    type="data",
                    array=monthly[("num_cited_urls", "sem")].fillna(0),
                    visible=True,
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average # URLs per Turn",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    output_dir = f"{OUTPUT_PATH}/{platform}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    file_name = "url_counts_over_time"
    if separate_cited_external_internal:
        file_name += "_split_cited"
    fig = with_paper_style(fig, config=styler(18, 20))
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")


def plot_retrieved_url_counts_over_time_by_model(platform="chatgpt"):
    df = _prepare_source_count_df(platform)
    if "model" not in df.columns:
        print(f"[{platform}] No `models` column in response_and_sources.pkl; skipping.")
        return
    df = df[df["model"].str.lower() != "unknown"].copy()
    df = df.dropna(subset=["month"])
    if len(df) == 0:
        return

    monthly = (
        df.groupby(["month", "model"])["num_retrieved_urls"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .sort_values(["model", "month"])
    )
    if len(monthly) == 0:
        return

    model_order = (
        monthly.groupby("model")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig = go.Figure()
    for model in model_order:
        model_df = monthly[monthly["model"] == model]
        if len(model_df) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=model_df["month"],
                y=model_df["mean"],
                mode="lines+markers",
                name=model,
                error_y=dict(
                    type="data",
                    array=model_df["sem"].fillna(0),
                    visible=True,
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average # Retrieved URLs per Turn",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    output_dir = f"{OUTPUT_PATH}/{platform}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    file_name = "retrieved_url_counts_over_time_by_model"
    fig = with_paper_style(fig, config=styler(18, 18), legend_pos=(0.8, 1.8))
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")


def plot_retrieved_cited_positions(separate_cited_external_internal=False):
    # ChatGPT-only: reads extract_retrieved_cited_source()'s output, which
    # has no Claude/Grok/DeepSeek equivalent (see save_topic_to_domains_json()'s
    # comment).
    os.makedirs(f"{OUTPUT_PATH}/chatgpt/{CONF}", exist_ok=True)
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/chatgpt/metadata/retrieved_cited_extracted_from_srcs.pkl"
    ).copy()

    def _valid_ref_index(value):
        try:
            rank = float(value)
        except (TypeError, ValueError):
            return np.nan
        if not np.isfinite(rank) or rank < 0:
            return np.nan
        # Source ranks are 0-indexed in metadata; shift by +1 for plotting so
        # log10(rank) starts at 0 instead of negative values.
        return rank + 1.0

    def _append_average_rank(rows, group_name, ranks):
        valid_ranks = [rank for rank in ranks if np.isfinite(rank)]
        if valid_ranks:
            rows.append(
                {
                    "group": group_name,
                    "average_rank": float(np.mean(valid_ranks)),
                }
            )

    plot_rows = []
    avg_rank_rows = []
    for _, row in df.iterrows():
        retrieved_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in row["srcs_retrieved"]
            if isinstance(item, dict) and item.get("url", "")
        }
        retrieved_ref_indices = []
        for item in row["srcs_retrieved"]:
            if not isinstance(item, dict):
                continue
            ref_index = _valid_ref_index(item.get("ref_index"))
            if not np.isfinite(ref_index):
                continue
            retrieved_ref_indices.append(ref_index)
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue
            plot_rows.append(
                {
                    "group": "Retrieved URLs",
                    "ref_index": ref_index,
                    "turn_index": turn_index,
                }
            )

        cited_ref_indices = []
        cited_external_ref_indices = []
        cited_internal_ref_indices = []
        for item in row["srcs_cited"]:
            if not isinstance(item, dict):
                continue
            ref_index = _valid_ref_index(item.get("ref_index"))
            if not np.isfinite(ref_index):
                continue
            if separate_cited_external_internal:
                cited_url = _normalize_url_for_source_matching(item.get("url", ""))
                is_external = bool(cited_url and cited_url in retrieved_urls)
                if is_external:
                    group_name = "Cited Retrieved/External URLs"
                    cited_external_ref_indices.append(ref_index)
                else:
                    group_name = "Cited Unexplained/Internal URLs"
                    cited_internal_ref_indices.append(ref_index)
            else:
                group_name = "Cited URLs"
                cited_ref_indices.append(ref_index)
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue
            plot_rows.append(
                {
                    "group": group_name,
                    "ref_index": ref_index,
                    "turn_index": turn_index,
                }
            )

        _append_average_rank(avg_rank_rows, "Retrieved URLs", retrieved_ref_indices)
        if separate_cited_external_internal:
            _append_average_rank(
                avg_rank_rows,
                "Cited Retrieved/External URLs",
                cited_external_ref_indices,
            )
            _append_average_rank(
                avg_rank_rows,
                "Cited Unexplained/Internal URLs",
                cited_internal_ref_indices,
            )
        else:
            _append_average_rank(avg_rank_rows, "Cited URLs", cited_ref_indices)

    plot_df = pd.DataFrame(plot_rows)
    avg_rank_df = pd.DataFrame(avg_rank_rows)
    if len(plot_df) == 0 and len(avg_rank_df) == 0:
        return

    if separate_cited_external_internal:
        rank_specs = [
            ("Retrieved URLs", "Retrieved", "#636EFA"),
            (
                "Cited Unexplained/Internal URLs",
                "Cited<br>Unexplained/Internal",
                "#E45756",
            ),
            (
                "Cited Retrieved/External URLs",
                "Cited<br>Retrieved/External",
                "#00CC96",
            ),
        ]
    else:
        rank_specs = [
            ("Retrieved URLs", "Retrieved", "#636EFA"),
            ("Cited URLs", "Cited", "#EF553B"),
        ]
    group_order = [group_name for group_name, _label, _color in rank_specs]

    if len(plot_df) > 0:
        aggregated_df = (
            plot_df.groupby(["group", "turn_index", "ref_index"])
            .size()
            .reset_index(name="count")
        )

        if len(group_order) == 1:
            offsets = [0.0]
        elif len(group_order) == 2:
            offsets = [-0.15, 0.15]
        elif len(group_order) == 3:
            offsets = [-0.2, 0.0, 0.2]
        else:
            offsets = np.linspace(-0.25, 0.25, len(group_order))
        x_offsets = {group_name: float(offset) for group_name, offset in zip(group_order, offsets)}

        fig = go.Figure()
        for group_name in group_order:
            subset = aggregated_df[aggregated_df["group"] == group_name]
            fig.add_trace(
                go.Scatter(
                    x=subset["turn_index"] + 1 + x_offsets[group_name],
                    y=subset["ref_index"],
                    mode="markers",
                    name=group_name,
                    marker=dict(
                        size=3,
                    ),
                    hovertemplate="Loop=%{x}<br>Rank=%{y}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Loop",
            yaxis_title="Rank",
        )
        fig.update_xaxes(dtick=5)
        fig.update_yaxes(autorange="reversed")
        file_name = "retrieved_cited_positions"
        if separate_cited_external_internal:
            file_name += "_split_cited"
        fig = with_paper_style(fig, config=styler(18, 16))
        fig.write_image(f"{OUTPUT_PATH}/chatgpt/{CONF}/{file_name}.pdf", format="pdf")

    if len(avg_rank_df) == 0:
        return

    violin_fig = go.Figure()
    use_log10_violin = separate_cited_external_internal
    if use_log10_violin:
        valid_values = pd.to_numeric(
            avg_rank_df["average_rank"], errors="coerce"
        ).to_numpy()
        valid_values = valid_values[np.isfinite(valid_values)]
        valid_values = valid_values[valid_values > 0]
        if len(valid_values) == 0:
            return
        valid_log_values = np.log10(valid_values)
        main_rank_min = float(np.floor(np.min(valid_log_values)))
        main_rank_max = float(np.ceil(np.max(valid_log_values)))
        if not np.isfinite(main_rank_min):
            main_rank_min = 0.0
        if not np.isfinite(main_rank_max):
            main_rank_max = 1.0
        if main_rank_max <= main_rank_min:
            main_rank_max = main_rank_min + 1.0
    else:
        main_rank_min = 0.0
        main_rank_max = float(avg_rank_df["average_rank"].max())
        if not np.isfinite(main_rank_max) or main_rank_max <= 0:
            main_rank_max = 1.0
    for group_name, label, color in rank_specs:
        subset = avg_rank_df.loc[
            avg_rank_df["group"] == group_name, "average_rank"
        ].dropna()
        if len(subset) == 0:
            continue

        zoom_percentile = 85 if ("Cited" in label and "Internal" in label) or label == "Cited" else 90
        if use_log10_violin:
            subset = subset[subset > 0]
            if len(subset) == 0:
                continue
            subset_plot = np.log10(subset.to_numpy())
        else:
            subset_plot = subset.to_numpy()

        violin_fig.add_trace(
            go.Violin(
                x=[label] * len(subset_plot),
                y=subset_plot,
                name=label,
                marker_color=color,
                line_color=color,
                box_visible=True,
                width=0.9,
                meanline_visible=True,
                showlegend=True,
                legendgroup=label,
            )
        )
        # mean_value = float(np.mean(subset_plot))
        # median_value = float(np.median(subset_plot))
        # annotation_text = f"mean={mean_value:,.1f}<br>median={median_value:,.1f}"
        # annotation_y = float(np.nanpercentile(subset_plot, zoom_percentile))
        # if not np.isfinite(annotation_y):
        #     annotation_y = mean_value
        # violin_fig.add_annotation(
        #     x=label,
        #     y=min(main_rank_max - 0.05, annotation_y),
        #     text=annotation_text,
        #     showarrow=True,
        #     arrowhead=1,
        #     ax=35,
        #     ay=-25,
        #     font=dict(size=12, color=color),
        #     bgcolor="rgba(255,255,255,0.85)",
        #     bordercolor=color,
        # )

    violin_fig.update_layout(
        xaxis_title="Source Type",
        yaxis_title="Average Rank (log10)" if use_log10_violin else "Average Rank",
        xaxis=dict(
            categoryorder="array",
            categoryarray=[label for _group_name, label, _color in rank_specs],
            tickangle=0,
        ),
        yaxis=dict(range=[main_rank_min, main_rank_max], autorange=False),
        # height=680,
        # width=900,
    )

    file_name = "retrieved_cited_positions_rank_violinplot"
    if separate_cited_external_internal:
        file_name += "_split_cited"
    violin_fig = with_paper_style(violin_fig, config=styler(18, 16), legend_pos=None)
    violin_fig.update_xaxes(tickangle=0, tickfont=dict(size=16))
    violin_fig.update_layout(
        # height=680,
        # width=900,
        yaxis=dict(range=[main_rank_min, main_rank_max], autorange=False),
    )
    violin_fig.write_image(f"{OUTPUT_PATH}/chatgpt/{CONF}/{file_name}.pdf", format="pdf")


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return []
        return parsed if isinstance(parsed, list) else []
    return []


def _as_valid_number(value, min_value=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(number):
        return np.nan
    if min_value is not None and number < min_value:
        return np.nan
    return number


def _build_retrieved_cited_rank_plot_rows(
    df,
    retrieved_rank_col,
    cited_rank_col,
    rank_transform=None,
    min_rank=None,
):
    rank_transform = rank_transform or (lambda rank: rank)
    plot_rows = []

    for _, row in df.iterrows():
        retrieved_sources = _as_list(row.get("srcs_retrieved", []))
        retrieved_ranks = _as_list(row.get(retrieved_rank_col, []))
        cited_sources = _as_list(row.get("srcs_cited", []))
        cited_ranks = _as_list(row.get(cited_rank_col, []))

        for idx, item in enumerate(retrieved_sources):
            if not isinstance(item, dict):
                continue
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue

            rank = np.nan
            if idx < len(retrieved_ranks):
                rank = rank_transform(_as_valid_number(retrieved_ranks[idx], min_rank))
            if np.isfinite(rank):
                plot_rows.append(
                    {
                        "group": "Retrieved URLs",
                        "rank": rank,
                        "turn_index": turn_index,
                    }
                )

        for idx, item in enumerate(cited_sources):
            if not isinstance(item, dict):
                continue
            turn_index = item.get("turn_index")
            if turn_index is None:
                continue
            rank = np.nan
            if idx < len(cited_ranks):
                rank = rank_transform(_as_valid_number(cited_ranks[idx], min_rank))
            if not np.isfinite(rank):
                continue
            plot_rows.append(
                {
                    "group": "Cited URLs",
                    "rank": rank,
                    "turn_index": turn_index,
                }
            )

    return plot_rows


def _plot_retrieved_cited_ranks(plot_rows, file_name, yaxis_title, platform="chatgpt"):
    plot_df = pd.DataFrame(plot_rows)
    if len(plot_df) == 0:
        return

    aggregated_df = (
        plot_df.groupby(["group", "turn_index", "rank"])
        .size()
        .reset_index(name="count")
    )

    x_offsets = {
        "Retrieved URLs": -0.1,
        "Cited URLs": 0.1,
    }

    fig = go.Figure()
    for group_name in ["Retrieved URLs", "Cited URLs"]:
        subset = aggregated_df[aggregated_df["group"] == group_name]
        fig.add_trace(
            go.Scatter(
                x=subset["turn_index"] + 1 + x_offsets[group_name],
                y=subset["rank"],
                mode="markers",
                name=group_name,
                marker=dict(
                    size=3,
                ),
                hovertemplate="Loop=%{x}<br>Rank=%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Loop",
        yaxis_title=yaxis_title,
    )
    fig.update_xaxes(dtick=5)
    fig.update_yaxes(autorange="reversed")
    output_dir = f"{OUTPUT_PATH}/{platform}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")


def plot_retrieved_cited_tranco_ranks(platform="chatgpt"):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_with_tranco_ranks.pkl"
    ).copy()
    plot_rows = _build_retrieved_cited_rank_plot_rows(
        df,
        retrieved_rank_col="ranks_srcs_retrieved",
        cited_rank_col="ranks_srcs_cited",
        min_rank=0,
    )
    _plot_retrieved_cited_ranks(
        plot_rows,
        file_name="retrieved_cited_tranco_ranks",
        yaxis_title="Tranco Rank",
        platform=platform,
    )

def evaluate_source_tranco_ranks(
    separate_cited_external_internal=False,
    grounding_level="turn",
    platform="chatgpt",
):
    # response_and_sources_with_tranco_ranks.pkl is a research-only artifact
    # (an external Tranco top-1M-domains ranking pass) not produced by
    # anything in this repo, for any platform -- same category of gap as
    # all_tools_categorized.json/web_calls_characterization.csv (see
    # README's "Pipeline Order & Known Gaps").
    grounding_level = _validated_grounding_level(grounding_level)
    os.makedirs(f"{OUTPUT_PATH}/{platform}/{CONF}", exist_ok=True)
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_with_tranco_ranks.pkl"
    ).copy()

    def _as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return []
        return []

    def _avg_valid_rank(ranks):
        ranks = _as_list(ranks)
        valid_ranks = []
        for rank in ranks:
            try:
                rank_value = float(rank)
            except (TypeError, ValueError):
                continue
            if rank_value > 0:
                valid_ranks.append(rank_value)
        if not valid_ranks:
            return np.nan
        return float(np.mean(valid_ranks))

    def _avg_valid_rank_by_mask(ranks, keep_mask):
        ranks = _as_list(ranks)
        valid_ranks = []
        for idx, rank in enumerate(ranks):
            if idx >= len(keep_mask) or not keep_mask[idx]:
                continue
            try:
                rank_value = float(rank)
            except (TypeError, ValueError):
                continue
            if rank_value > 0:
                valid_ranks.append(rank_value)
        if not valid_ranks:
            return np.nan
        return float(np.mean(valid_ranks))

    df["retrieved_avg_rank"] = df["ranks_srcs_retrieved"].apply(_avg_valid_rank)

    if separate_cited_external_internal:
        cited_external_avg = {}
        cited_internal_avg = {}
        previous_retrieved_by_conv = {}
        if grounding_level == "conversation":
            sort_cols = [col for col in ["conv_id", "turn_id", "time"] if col in df.columns]
            if sort_cols:
                iterable_df = df.sort_values(sort_cols, kind="stable")
            else:
                iterable_df = df
        else:
            iterable_df = df

        for row_index, row_data in iterable_df.iterrows():
            row = type("Row", (), row_data.to_dict())()
            retrieved_urls = _row_retrieved_urls_by_grounding(
                row,
                previous_retrieved_by_conv,
                grounding_level,
            )
            cited_sources = _as_list(getattr(row, "srcs_cited", []))
            external_mask = []
            internal_mask = []
            for item in cited_sources:
                if not isinstance(item, dict):
                    external_mask.append(False)
                    internal_mask.append(False)
                    continue
                cited_url = _normalize_url_for_source_matching(item.get("url", ""))
                is_external = bool(cited_url and cited_url in retrieved_urls)
                external_mask.append(is_external)
                internal_mask.append(not is_external)

            cited_external_avg[row_index] = (
                _avg_valid_rank_by_mask(
                    getattr(row, "ranks_srcs_cited", []),
                    external_mask,
                )
            )
            cited_internal_avg[row_index] = (
                _avg_valid_rank_by_mask(
                    getattr(row, "ranks_srcs_cited", []),
                    internal_mask,
                )
            )

        df["cited_external_avg_rank"] = df.index.map(cited_external_avg.get)
        df["cited_internal_avg_rank"] = df.index.map(cited_internal_avg.get)
        rank_specs = [
            ("retrieved_avg_rank", "Search Results", "#636EFA"),
            ("cited_external_avg_rank", "Cited<br>Search Results", "#00CC96"),
            ("cited_internal_avg_rank", "Cited<br>Parametric", "#E45756"),
        ]
    else:
        df["cited_avg_rank"] = df["ranks_srcs_cited"].apply(_avg_valid_rank)
        rank_specs = [
            ("retrieved_avg_rank", "Search Results", "#636EFA"),
            ("cited_avg_rank", "Cited", "#EF553B"),
        ]

    print("Average Tranco rank ranges:")
    for col, label, _color in rank_specs:
        subset = df[col].dropna()
        if len(subset) == 0:
            print(f"{label}: no valid ranks")
        else:
            print(f"{label}: {subset.min():,.0f}-{subset.max():,.0f}")

    box_df = df[[col for col, _label, _color in rank_specs]].copy()
    box_fig = go.Figure()

    all_positive_values = []
    for col, _label, _color in rank_specs:
        col_values = pd.to_numeric(box_df[col], errors="coerce").to_numpy()
        col_values = col_values[np.isfinite(col_values)]
        col_values = col_values[col_values > 0]
        if len(col_values) > 0:
            all_positive_values.append(col_values)

    if len(all_positive_values) > 0:
        combined_values = np.concatenate(all_positive_values)
        global_min_log = float(np.floor(np.log10(np.min(combined_values))))
        global_max_log = float(np.ceil(np.log10(np.max(combined_values))))
        if global_max_log <= global_min_log:
            global_max_log = global_min_log + 1.0
    else:
        global_min_log = 0.0
        global_max_log = 1.0

    for col, label, color in rank_specs:
        subset = box_df[col].dropna()
        subset = subset[subset > 0]
        if len(subset) == 0:
            continue
        subset_log = np.log10(subset.to_numpy())
        box_fig.add_trace(
            go.Violin(
                x=[label] * len(subset_log),
                y=subset_log,
                name=label,
                marker_color=color,
                line_color=color,
                width=0.9,
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                legendgroup=label,
            )
        )

    box_fig.update_layout(
        xaxis_title="Source Type",
        xaxis=dict(tickangle=0),
        yaxis_title="Average Rank (log10)",
        yaxis=dict(range=[global_min_log, global_max_log], tickmode="linear", dtick=1),
        violinmode="group",
        margin=dict(t=5, b=130, r=5),
    )
    violin_file_name = "source_rank_violinplot"
    if separate_cited_external_internal:
        violin_file_name += f"_split_cited_{grounding_level}_grounding"
    box_fig = with_paper_style(box_fig, config=styler(26, 16), legend_pos=None)
    box_fig.update_xaxes(tickangle=0, tickfont=dict(size=26))
    box_fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{violin_file_name}.pdf", format="pdf")

    paired_fig = make_subplots(rows=1, cols=1)
    if separate_cited_external_internal:
        scatter_specs = [
            ("cited_external_avg_rank", "Cited Search Results", "#00CC96"),
            ("cited_internal_avg_rank", "Cited Parametric", "#E45756"),
        ]
    else:
        scatter_specs = [
            ("cited_avg_rank", "Cited", "#00CC96"),
        ]

    diagonal_min = np.inf
    diagonal_max = -np.inf
    has_points = False
    for y_col, label, color in scatter_specs:
        subset = df[["retrieved_avg_rank", y_col]].dropna()
        if len(subset) == 0:
            continue
        has_points = True
        diagonal_min = min(
            diagonal_min,
            float(subset["retrieved_avg_rank"].min()),
            float(subset[y_col].min()),
        )
        diagonal_max = max(
            diagonal_max,
            float(subset["retrieved_avg_rank"].max()),
            float(subset[y_col].max()),
        )
        paired_fig.add_trace(
            go.Scatter(
                x=subset["retrieved_avg_rank"],
                y=subset[y_col],
                mode="markers",
                name=label,
                marker=dict(color=color, size=8),
                showlegend=separate_cited_external_internal,
            ),
            row=1,
            col=1,
        )

    if has_points:
        if diagonal_min == diagonal_max:
            diagonal_min -= 1
            diagonal_max += 1
        paired_fig.add_trace(
            go.Scatter(
                x=[diagonal_min, diagonal_max],
                y=[diagonal_min, diagonal_max],
                mode="lines",
                line=dict(color="black", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        paired_fig.update_xaxes(
            title_text="Search Results Avg Rank",
            range=[diagonal_min, diagonal_max],
            row=1,
            col=1,
        )
        y_title = "Cited Avg Rank"
        if separate_cited_external_internal:
            y_title = "Cited Avg Rank (External/Internal)"
        paired_fig.update_yaxes(
            title_text=y_title,
            range=[diagonal_min, diagonal_max],
            row=1,
            col=1,
        )

    paired_file_name = "source_rank_paired_plot"
    if separate_cited_external_internal:
        paired_file_name += f"_split_cited_{grounding_level}_grounding"
    legend_pos = (0.98, 1.2) if separate_cited_external_internal else None
    paired_fig = with_paper_style(
        paired_fig,
        config=styler(18, 16),
        legend_pos=legend_pos,
    )
    paired_fig.update_layout(width=700 if separate_cited_external_internal else 500, height=400)
    paired_fig.write_image(f"{OUTPUT_PATH}/{platform}/{CONF}/{paired_file_name}.pdf", format="pdf")

def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")  # fallback
    

if __name__ == "__main__":
    # extract_retrieved_cited_source reads ChatGPT's raw turn_msgs wire
    # format directly -- no Claude/Grok/DeepSeek equivalent (see
    # plot_retrieved_cited_positions' comment below), so this step stays a
    # single ChatGPT-only call rather than a platform loop.
    web_df = load_web_data_from_file(fmt="pkl", platform="chatgpt")
    print(f"Loaded web data: {len(web_df)}")
    extract_retrieved_cited_source(web_df)

    # Per-platform analyses: each writes under its own
    # outputs/<platform>/source_selection/ so results from different
    # platforms never overwrite each other.
    for platform in PLATFORMS:
        plot_url_counts_over_time(separate_cited_external_internal=True, platform=platform)
        plot_retrieved_url_counts_over_time_by_model(platform=platform)
        plot_top_domains(
            separate_cited_external_internal=True,
            grounding_level="conversation",
            platform=platform,
        )
        try:
            evaluate_source_tranco_ranks(
                separate_cited_external_internal=True,
                grounding_level="conversation",
                platform=platform,
            )
        except FileNotFoundError as e:
            # response_and_sources_with_tranco_ranks.pkl is a research-only
            # artifact not produced by anything in this repo, for any
            # platform -- see evaluate_source_tranco_ranks' own comment.
            print(f"[{platform}] Skipping evaluate_source_tranco_ranks -- {e}")

    # plot_retrieved_cited_positions reads extract_retrieved_cited_source's
    # ChatGPT-only output above -- no Claude/Grok/DeepSeek equivalent, so
    # this also stays a single ChatGPT-only call.
    plot_retrieved_cited_positions(separate_cited_external_internal=True)

    # Combined/cross-platform: each of these already loops over all 4
    # platforms internally (or is inherently ChatGPT-model-specific), so
    # they run once, not per platform. plot_top_domains must have already
    # run for every platform above -- evaluate_unique_retrieved_domains_by_
    # platform() reads its retrieved_domains_sorted.csv output for each.
    compute_average_citations_and_retrievals_per_response(grounding_level="conversation")
    compute_average_citations_and_retrievals_per_response_by_model(
        grounding_level="conversation"
    )
    plot_retrieved_urls_per_web_query_histogram()
    evaluate_unique_retrieved_domains_by_platform()

    pass
