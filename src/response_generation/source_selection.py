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
from src.web_search_decision.chatgpt_extraction import load_web_data_from_file
from src.response_generation.response_generation import _load_response_source_similarity_input
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


def extract_retrieved_safe_cited_source(web_df):
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
    web_df["srcs_safe_urls"] = [{}] * len(web_df)
    web_df["srcs_cited"] = [{}] * len(web_df)

    for i, row in tqdm(web_df.iterrows()):
        msgs = json.loads(row['turn_msgs'])
        srcs_retrieved = []
        srcs_safe_urls = []
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

            # safe urls
            safe_urls = msg.get('metadata', {}).get('safe_urls', [])
            for r in safe_urls:
                if r:
                    url = r.removesuffix("?utm_source=chatgpt.com").removesuffix("&utm_source=chatgpt.com")
                    d = urlparse(url).netloc.replace("www.", "")
                    srcs_safe_urls.append({
                        "url": url,
                        "domain": d
                    })

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
        web_df.at[i, "srcs_safe_urls"] = srcs_safe_urls
        web_df.at[i, "srcs_cited"] = _dedupe_cited_items(srcs_cited)

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)
    
    web_df.to_csv(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.csv",
        index=False,
    )
    web_df.to_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
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
    """Load outputs/[<model>/]metadata/response_and_sources.pkl and add
    per-row retrieved/safe/cited URL counts.

    Pipeline dependency: this file is NOT produced by anything in this
    module. Run response_generation.extract_response_and_sources(web_df)
    (or extract_response_and_sources_other_platforms() for non-ChatGPT
    platforms) first -- it's a separate scrape/dedupe pass over the raw
    turn messages, distinct from extract_retrieved_safe_cited_source()
    above, which writes a differently-named file
    (retrieved_safe_cited_extracted_from_srcs.pkl) that functions in *this*
    module read instead. The two pipelines are not interchangeable inputs
    for each other despite the similar-sounding names.
    """
    if model == "chatgpt" or model == "":
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/metadata/response_and_sources.pkl"
        ).copy()
    else:
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/{model}/metadata/response_and_sources.pkl"
        ).copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()
    if "openai_models" in df.columns:
        df["model"] = df["openai_models"].apply(_primary_model)
    df["num_retrieved_urls"] = df["srcs_retrieved"].apply(_unique_source_count)
    df["num_safe_urls"] = df["srcs_safe_urls"].apply(_unique_source_count)
    df["num_cited_urls"] = df["srcs_cited"].apply(_unique_source_count)
    return df


def count_unique_retrieved_safe_cited():
    """Count of responses with at least one retrieved/safe/cited URL each.
    Requires response_generation.extract_response_and_sources() to have
    been run first -- see _prepare_source_count_df()'s docstring."""
    df = _prepare_source_count_df()
    return {
        "retrieved_urls": (df["num_retrieved_urls"] > 0).sum(),
        "safe_urls": (df["num_safe_urls"] > 0).sum(),
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
            f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl",
            f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.csv",
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
            f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.pkl",
            f"{OUTPUT_PATH}/metadata/query_reformulation_with_thought_src_mem.csv",
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
):
    query_count_lookup = _load_query_count_lookup("chatgpt")
    query_groups_lookup = _load_query_groups_lookup("chatgpt")
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

    for model in ["chatgpt", "claude", "grok", "deepseek"]:
        print(model)
        df = _prepare_source_count_df(model)
        _print_source_count_summaries_for_df(
            df,
            unique=unique,
            grounding_level=grounding_level,
        )


def compute_average_citations_and_retrievals_per_response_by_openai_model(
    unique=False,
    grounding_level="turn",
):
    grounding_level = _validated_grounding_level(grounding_level)
    df = _prepare_source_count_df("chatgpt").copy()
    df["model"] = df["openai_models"].apply(_primary_model)
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
    os.makedirs(f"{OUTPUT_PATH}/{CONF}", exist_ok=True)

    for model in ["chatgpt", "claude", "grok", "deepseek"]:
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

        file_name = f"{model}_retrieved_urls_per_web_query_histogram"
        html_path = f"{OUTPUT_PATH}/{CONF}/{file_name}.html"
        pdf_path = f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf"
        print(f"Writing histogram HTML: {html_path}")
        fig.write_html(html_path)
        fig = with_paper_style(fig, config=styler(18, 16), legend_pos=None)
        try:
            print(f"Writing histogram PDF: {pdf_path}")
            fig.write_image(pdf_path, format="pdf")
        except Exception as exc:
            print(f"Failed to write histogram PDF for {model}: {exc}")


def compute_subset_counts(df, retrieved_col, safe_col, cited_col, key="url"):
    counts = {}

    for _, row in df.iterrows():
        retrieved = {
            item.get(key, "")
            for item in row[retrieved_col]
            if isinstance(item, dict) and item.get(key, "")
        }
        safe = {
            item.get(key, "")
            for item in row[safe_col]
            if isinstance(item, dict) and item.get(key, "")
        }
        cited = {
            item.get(key, "")
            for item in row[cited_col]
            if isinstance(item, dict) and item.get(key, "")
        }

        safe_in_retrieved = safe.issubset(retrieved)
        cited_in_safe = cited.issubset(safe)
        cited_in_retrieved = cited.issubset(retrieved)

        condition = (safe_in_retrieved, cited_in_safe, cited_in_retrieved)
        counts[condition] = counts.get(condition, 0) + 1

    return counts


def plot_subset_condition_counts():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()

    def _compute_cited_subset_counts(df, retrieved_col, cited_col, key="url"):
        counts = {False: 0, True: 0}
        for _, row in df.iterrows():
            retrieved = {
                item.get(key, "")
                for item in row[retrieved_col]
                if isinstance(item, dict) and item.get(key, "")
            }
            cited = {
                item.get(key, "")
                for item in row[cited_col]
                if isinstance(item, dict) and item.get(key, "")
            }
            counts[cited.issubset(retrieved)] += 1
        return counts

    url_counts = _compute_cited_subset_counts(
        df, "srcs_retrieved", "srcs_cited", key="url"
    )
    domain_counts = _compute_cited_subset_counts(
        df, "srcs_retrieved", "srcs_cited", key="domain"
    )

    label_map = {
        False: "C⊄R",
        True: "C⊆R",
    }

    rows = []
    for condition, label in [(False, label_map[False]), (True, label_map[True])]:
        rows.append(
            {"Condition": label, "Count": url_counts.get(condition, 0), "Type": "URLs"}
        )
        rows.append(
            {
                "Condition": label,
                "Count": domain_counts.get(condition, 0),
                "Type": "Domains",
            }
        )

    plot_df = pd.DataFrame(rows)
    order = (
        plot_df.groupby("Condition")["Count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    plot_df["Condition"] = pd.Categorical(
        plot_df["Condition"], categories=order, ordered=True
    )
    plot_df = plot_df.sort_values("Condition")

    fig = go.Figure()
    for source_type in ["URLs", "Domains"]:
        subset = plot_df[plot_df["Type"] == source_type]
        fig.add_trace(
            go.Bar(
                x=subset["Condition"],
                y=subset["Count"],
                name=source_type,
                text=subset["Count"],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Subset Conditions",
        yaxis_title="#Turns",
    )
    file_name = "subset_relations_urls_and_domains"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18))
    fig.update_xaxes(tickfont=dict(size=14))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def save_topic_to_domains_json():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()

    topic_to_domains = {}
    for _, row in df.iterrows():
        topic = row.get("topic", "Other")
        if pd.isna(topic):
            topic = "Other"
        if str(topic).strip().lower() == "other":
            continue

        if topic not in topic_to_domains:
            topic_to_domains[topic] = set()

        topic_to_domains[topic].update(
            item.get("domain", "")
            for item in row["srcs_retrieved"]
            if isinstance(item, dict) and item.get("domain", "")
        )
        topic_to_domains[topic].update(
            item.get("domain", "")
            for item in row["srcs_safe_urls"]
            if isinstance(item, dict) and item.get("domain", "")
        )
        topic_to_domains[topic].update(
            item.get("domain", "")
            for item in row["srcs_cited"]
            if isinstance(item, dict) and item.get("domain", "")
        )

    topic_to_domains = {
        topic: sorted(values)
        for topic, values in topic_to_domains.items()
    }

    to_json(
        topic_to_domains,
        f"{OUTPUT_PATH}/metadata/topic_to_domains.json",
    )


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
            f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
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
    # fig.write_html(f"{output_dir}/{file_name}.html")
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




def _user_prompt_domain_counter(top_k=20):
    import tldextract
    for platform in ["", "claude", "grok", "deepseek"]:
        print(platform)
        df = _load_domain_plot_df(platform)
        counter = Counter()

        for _, row in df.iterrows():

            history = row["user_msg_history"]
            if not history:
                continue

            prompt = str(history[-1]).lower()

            domains = set()

            ###############################
            # Retrieved
            ###############################
            for url in row.get("srcs_retrieved", []):
                dom = tldextract.extract(url["url"])
                if dom.domain:
                    domains.add(dom.domain.lower())

            ###############################
            # Cited
            ###############################
            for url in row.get("srcs_cited", []):
                dom = tldextract.extract(url["url"])
                if dom.domain:
                    domains.add(dom.domain.lower())

            ###############################
            # Check whether user mentioned it
            ###############################
            for d in domains:
                if re.search(rf"\b{re.escape(d)}\b", prompt, flags=re.IGNORECASE):
                    counter[_normalize_domain_for_top_plots(d)] += 1

        out = (
            pd.DataFrame(counter.items(), columns=["domain", "count"])
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

        total = out["count"].sum()

        out["percentage"] = 100 * out["count"] / total

        print(out.head(top_k))


def plot_top_domains(
    separate_cited_external_internal=False,
    grounding_level="turn",
    top_k=10,
):
    grounding_level = _validated_grounding_level(grounding_level)
    platform = "deepseek"
    safe_platform = platform if platform else "chatgpt"
    df = _load_domain_plot_df(platform=platform)
    output_dir = f"{OUTPUT_PATH}/{CONF}"
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
        f"{output_dir}/{safe_platform}_retrieved_domains_sorted.csv",
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
                f"{safe_platform}_cited_domains_split_{grounding_level}_grounding_sorted.csv"
            ),
            index=False,
        )
        split_df[split_df["external_count"] > 0][["domain", "external_count"]].sort_values(
            "external_count",
            ascending=False,
        ).to_csv(
            (
                f"{output_dir}/"
                f"{safe_platform}_cited_external_domains_{grounding_level}_grounding_sorted.csv"
            ),
            index=False,
        )
        split_df[split_df["internal_count"] > 0][["domain", "internal_count"]].sort_values(
            "internal_count",
            ascending=False,
        ).to_csv(
            (
                f"{output_dir}/"
                f"{safe_platform}_cited_internal_domains_{grounding_level}_grounding_sorted.csv"
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
            f"{output_dir}/{safe_platform}_cited_domains_sorted.csv",
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
        f"top_domains_overall_split_cited_{grounding_level}_grounding_{platform}"
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


def _plot_top_domains_for_subset(
    df,
    subset_label,
    output_dir,
    separate_cited_external_internal=False,
    use_plot_top_domains_setup=False,
):
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
        vertical_spacing=0.3,
    )

    has_data = False
    retrieved_all_df = _domain_counter(df, "srcs_retrieved", top_k=None)
    retrieved_denominator = float(retrieved_all_df["count"].sum()) if len(retrieved_all_df) > 0 else 0.0
    retrieved_df = retrieved_all_df.head(10)
    if retrieved_denominator > 0 and len(retrieved_df) > 0:
        has_data = True
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
        split_df = _cited_domain_counter_split(df, top_k=None)
        cited_external_denominator = (
            float(split_df["external_count"].sum()) if len(split_df) > 0 else 0.0
        )
        cited_internal_denominator = (
            float(split_df["internal_count"].sum()) if len(split_df) > 0 else 0.0
        )

        external_df = split_df[split_df["external_count"] > 0].copy()
        external_df = external_df.sort_values("external_count", ascending=False).head(20)
        if cited_external_denominator > 0 and len(external_df) > 0:
            has_data = True
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
        internal_df = internal_df.sort_values("internal_count", ascending=False).head(20)
        if cited_internal_denominator > 0 and len(internal_df) > 0:
            has_data = True
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
        cited_denominator = float(cited_all_df["count"].sum()) if len(cited_all_df) > 0 else 0.0
        cited_df = cited_all_df.head(10)
        if cited_denominator > 0 and len(cited_df) > 0:
            has_data = True
            fig.add_trace(
                go.Bar(
                    x=cited_df["domain"],
                    y=cited_df["count"] / cited_denominator,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
    if not has_data:
        return

    fig.update_layout(
        # height=1000,
        margin=dict(l=70, b=60, t=30, r=40),
    )
    fig.update_xaxes(
        tickangle=-20,
        automargin=True,
    )
    fig.add_annotation(
        x=-0.12,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Percentage of URLs",
        textangle=-90,
        showarrow=False,
        font=dict(size=18, color="black"),
    )
    fig.update_yaxes(tickformat=".0%")
    file_name = (
        "top_20_domains_split_cited"
        if separate_cited_external_internal
        else "top_20_domains"
    )
    x_tickfont_size = None
    if use_plot_top_domains_setup:
        x_tickfont_size = 16
    _write_figure(
        fig,
        output_dir,
        file_name,
        styler(18, 18),
        legend_pos=None,
        x_tickfont_size=x_tickfont_size,
        y_tickfont_size=14,
    )


def plot_top_domains_by_selected_topics(separate_cited_external_internal=False):
    df = _load_domain_plot_df()
    # selected_topics = ["Health", "Travel", "Finance", "Politics & History", "Science"]
    selected_topics = set(df["topic"].unique())

    for topic in selected_topics:
        topic_df = df[df["topic"] == topic].copy()
        if len(topic_df) == 0:
            continue
        safe_topic = re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_").lower()
        output_dir = f"{OUTPUT_PATH}/{CONF}/top_domains_by_topic/{safe_topic}"
        _plot_top_domains_for_subset(
            topic_df,
            topic,
            output_dir,
            separate_cited_external_internal=separate_cited_external_internal,
            use_plot_top_domains_setup=True,
        )


def plot_top_domains_by_model(separate_cited_external_internal=False):
    df = _load_domain_plot_df()
    df["model"] = df["openai_models"].apply(_primary_model)
    df = df[df["model"].str.lower() != "unknown"].copy()

    for model in sorted(df["model"].dropna().unique()):
        model_df = df[df["model"] == model].copy()
        if len(model_df) == 0:
            continue
        safe_model = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
        output_dir = f"{OUTPUT_PATH}/{CONF}/top_domains_by_model/{safe_model}"
        _plot_top_domains_for_subset(
            model_df,
            model,
            output_dir,
            separate_cited_external_internal=separate_cited_external_internal,
        )


def _load_domain_rank_lookup(platform):
    path = f"{OUTPUT_PATH}/{CONF}/{platform}_retrieved_domains_sorted.csv"
    df = pd.read_csv(path).copy()
    if "domain" not in df.columns:
        raise ValueError(f"Missing `domain` column in {path}")
    df = df.dropna(subset=["domain"]).copy()
    df["domain"] = df["domain"].astype(str).map(_normalize_domain_for_top_plots)
    df = df[df["domain"] != ""].copy().reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return dict(zip(df["domain"], df["rank"])), df


def plot_chatgpt_claude_retrieved_domain_rank_scatter(top_k=100):
    chatgpt_rank_lookup, chatgpt_df = _load_domain_rank_lookup("chatgpt")
    claude_rank_lookup, claude_df = _load_domain_rank_lookup("claude")

    chatgpt_top = chatgpt_df.head(top_k).copy()
    chatgpt_top["other_rank"] = chatgpt_top["domain"].map(claude_rank_lookup)

    claude_top = claude_df.head(top_k).copy()
    claude_top["other_rank"] = claude_top["domain"].map(chatgpt_rank_lookup)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"ChatGPT Top {top_k}: Claude Rank",
            f"Claude Top {top_k}: ChatGPT Rank",
        ],
        horizontal_spacing=0.14,
    )

    fig.add_trace(
        go.Scatter(
            x=chatgpt_top["rank"],
            y=chatgpt_top["other_rank"],
            mode="markers",
            marker=dict(size=8, color="#3765E5"),
            text=chatgpt_top["domain"],
            hovertemplate="Domain=%{text}<br>ChatGPT rank=%{x}<br>Claude rank=%{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=claude_top["rank"],
            y=claude_top["other_rank"],
            mode="markers",
            marker=dict(size=8, color="#E45756"),
            text=claude_top["domain"],
            hovertemplate="Domain=%{text}<br>Claude rank=%{x}<br>ChatGPT rank=%{y}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    max_rank = max(len(chatgpt_df), len(claude_df))
    fig.update_xaxes(title_text="Source Rank", row=1, col=1)
    fig.update_yaxes(title_text="Other Platform Rank", row=1, col=1)
    fig.update_xaxes(title_text="Source Rank", row=1, col=2)
    fig.update_yaxes(title_text="Other Platform Rank", row=1, col=2)
    fig.update_xaxes(range=[0, top_k + 1])
    fig.update_yaxes(range=[0, max_rank + 1], autorange="reversed")
    fig.update_layout(
        margin=dict(l=70, b=70, t=60, r=30),
    )

    output_dir = f"{OUTPUT_PATH}/{CONF}"
    file_name = "chatgpt_claude_retrieved_domain_rank_scatter"
    fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 10), legend_pos=None)
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

    chatgpt_top.to_csv(
        f"{output_dir}/chatgpt_top_{top_k}_domains_with_claude_ranks.csv",
        index=False,
    )
    claude_top.to_csv(
        f"{output_dir}/claude_top_{top_k}_domains_with_chatgpt_ranks.csv",
        index=False,
    )


def evaluate_unique_retrieved_domains_by_platform():
    platforms = ["chatgpt", "claude", "grok", "deepseek"]
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)

    platform_dfs = {}
    platform_sets = {}
    for platform in platforms:
        path = f"{output_dir}/{platform}_retrieved_domains_sorted.csv"
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


def compare_domain_reliability_by_source_type(model_name="gpt-4o-mini", temperature=0.0):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()
    results_path = (
        f"{OUTPUT_PATH}/{CONF}/domain_reliability_evaluation/"
        f"{model_name}/{temperature}/results.json"
    )
    reliability_results = load_json(results_path)

    reliability_lookup = {}
    for topic, topic_results in reliability_results.items():
        reliability_lookup[topic] = {}
        for domain, raw_result in topic_results.items():
            score = None
            if isinstance(raw_result, str) and raw_result.strip():
                try:
                    score = json.loads(raw_result).get("reliability_score")
                except json.JSONDecodeError:
                    score = None
            elif isinstance(raw_result, dict):
                if "reliability_score" in raw_result:
                    score = raw_result.get("reliability_score")
                elif "Output" in raw_result and raw_result["Output"]:
                    try:
                        score = json.loads(raw_result["Output"]).get("reliability_score")
                    except json.JSONDecodeError:
                        score = None
            if score is not None:
                reliability_lookup[topic][domain] = float(score)

    def _avg_score(row, col_name):
        topic_scores = reliability_lookup.get(row["topic"], {})
        domains = sorted(
            {
                item.get("domain", "")
                for item in row[col_name]
                if isinstance(item, dict) and item.get("domain", "")
            }
        )
        scores = [topic_scores[d] for d in domains if d in topic_scores]
        if not scores:
            return np.nan
        return float(np.mean(scores))

    df["retrieved_reliability"] = df.apply(
        lambda row: _avg_score(row, "srcs_retrieved"), axis=1
    )
    df["cited_reliability"] = df.apply(
        lambda row: _avg_score(row, "srcs_cited"), axis=1
    )

    plot_df = df.dropna(subset=["retrieved_reliability", "cited_reliability"]).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=plot_df["retrieved_reliability"],
            name="Retrieved",
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Violin(
            y=plot_df["cited_reliability"],
            name="Cited",
            box_visible=True,
            meanline_visible=True,
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Source Type",
        yaxis_title="Average Domain Reliability",
    )
    fig.update_yaxes(range=[1, 5])

    file_name = "domain_reliability_by_source_type"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 14))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    ttest_results = {}
    comparisons = [
        ("retrieved_reliability", "cited_reliability", "retrieved_vs_cited"),
    ]
    for left_col, right_col, label in comparisons:
        paired = plot_df[[left_col, right_col]].dropna()
        if len(paired) == 0:
            continue
        stat, pvalue = ttest_rel(paired[left_col], paired[right_col])
        ttest_results[label] = {
            "n": int(len(paired)),
            "mean_left": float(paired[left_col].mean()),
            "mean_right": float(paired[right_col].mean()),
            "t_statistic": float(stat),
            "p_value": float(pvalue),
        }

    to_json(
        ttest_results,
        f"{OUTPUT_PATH}/{CONF}/{file_name}_paired_ttests.json",
    )
    plot_df[
        [
            "user_id",
            "conv_id",
            "turn_id",
            "topic",
            "retrieved_reliability",
            "cited_reliability",
        ]
    ].to_csv(
        f"{OUTPUT_PATH}/{CONF}/{file_name}_samples.csv",
        index=False,
    )

    return ttest_results


def plot_url_counts_over_time(separate_cited_external_internal=False):
    df = _prepare_source_count_df()

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
    file_name = "url_counts_over_time"
    if separate_cited_external_internal:
        file_name += "_split_cited"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 20))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def plot_grounding_rate_violin_over_time():
    df = _prepare_source_count_df()

    rows = []
    for _, row in df.iterrows():
        retrieved_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in row.get("srcs_retrieved", [])
            if isinstance(item, dict) and item.get("url", "")
        }
        cited_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in row.get("srcs_cited", [])
            if isinstance(item, dict) and item.get("url", "")
        }
        cited_urls = {url for url in cited_urls if url}
        if len(cited_urls) == 0:
            continue

        cited_external_urls = cited_urls & retrieved_urls
        cited_internal_urls = cited_urls - retrieved_urls
        grounding_rate = len(cited_external_urls) / len(cited_urls)
        unexplained_rate = len(cited_internal_urls) / len(cited_urls)

        rows.append(
            {
                "user_id": row.get("user_id"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "time": row.get("time"),
                "month": row.get("month"),
                "num_cited_urls": int(len(cited_urls)),
                "num_cited_external_urls": int(len(cited_external_urls)),
                "num_cited_internal_urls": int(len(cited_internal_urls)),
                "grounding_rate": float(grounding_rate),
                "unexplained_rate": float(unexplained_rate),
            }
        )

    plot_df = pd.DataFrame(rows)
    if len(plot_df) == 0:
        return plot_df

    plot_df["month"] = pd.to_datetime(plot_df["month"], errors="coerce")
    plot_df = plot_df.dropna(subset=["month"]).sort_values("month")
    if len(plot_df) == 0:
        return plot_df

    plot_df["month_label"] = plot_df["month"].dt.strftime("%b %Y")
    month_order = (
        plot_df[["month", "month_label"]]
        .drop_duplicates()
        .sort_values("month")["month_label"]
        .tolist()
    )
    tickvals_every_2_months = month_order[::2] if month_order else []
    shown_months = set(tickvals_every_2_months)
    plot_df = plot_df[plot_df["month_label"].isin(shown_months)].copy()
    if len(plot_df) == 0:
        return plot_df
    shown_month_order = [m for m in month_order if m in shown_months]
    month_to_pos = {m: float(i) for i, m in enumerate(shown_month_order)}
    plot_df["month_pos"] = plot_df["month_label"].map(month_to_pos)
    x_offset = 0.15

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=plot_df["month_pos"] - x_offset,
            y=plot_df["grounding_rate"],
            name="Grounding Rate (Cited Retrieved/External)",
            marker_color="#00CC96",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            width=0.42,
        )
    )
    fig.add_trace(
        go.Violin(
            x=plot_df["month_pos"] + x_offset,
            y=plot_df["unexplained_rate"],
            name="Unexplained Rate (Cited Internal)",
            marker_color="#E45756",
            box_visible=True,
            meanline_visible=True,
            showlegend=True,
            width=0.42,
        )
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Grounding Rate",
        xaxis=dict(
            tickmode="array",
            tickvals=[month_to_pos[m] for m in shown_month_order],
            ticktext=shown_month_order,
            tickangle=-30,
            range=[-0.6, max(len(shown_month_order) - 0.4, 0.6)],
        ),
        violinmode="overlay",
        margin=dict(b=120),
    )
    fig.update_yaxes(tickformat=".0%")

    file_name = "grounding_rate_violin_over_time"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_xaxes(
        tickangle=-30,
        tickmode="array",
        tickvals=[month_to_pos[m] for m in shown_month_order],
        ticktext=shown_month_order,
    )
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    monthly_stats = (
        plot_df.groupby(["month", "month_label"])
        .agg(
            grounding_rate_mean=("grounding_rate", "mean"),
            grounding_rate_std=("grounding_rate", "std"),
            unexplained_rate_mean=("unexplained_rate", "mean"),
            unexplained_rate_std=("unexplained_rate", "std"),
            num_turns=("grounding_rate", "count"),
        )
        .reset_index()
        .sort_values("month")
    )
    grounding_std = monthly_stats["grounding_rate_std"].fillna(0)
    unexplained_std = monthly_stats["unexplained_rate_std"].fillna(0)
    grounding_lower = (
        monthly_stats["grounding_rate_mean"] - grounding_std
    ).clip(lower=0, upper=1)
    grounding_upper = (
        monthly_stats["grounding_rate_mean"] + grounding_std
    ).clip(lower=0, upper=1)
    unexplained_lower = (
        monthly_stats["unexplained_rate_mean"] - unexplained_std
    ).clip(lower=0, upper=1)
    unexplained_upper = (
        monthly_stats["unexplained_rate_mean"] + unexplained_std
    ).clip(lower=0, upper=1)

    mean_std_fig = go.Figure()
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=grounding_lower,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=grounding_upper,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 204, 150, 0.2)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=monthly_stats["grounding_rate_mean"],
            mode="lines+markers",
            name="Grounding Rate (Cited Retrieved/External)",
            marker_color="#00CC96",
            line=dict(color="#00CC96"),
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=unexplained_lower,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=unexplained_upper,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(228, 87, 86, 0.2)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    mean_std_fig.add_trace(
        go.Scatter(
            x=monthly_stats["month_label"],
            y=monthly_stats["unexplained_rate_mean"],
            mode="lines+markers",
            name="Unexplained Rate (Cited Internal)",
            marker_color="#E45756",
            line=dict(color="#E45756"),
        )
    )
    mean_std_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Rate",
        xaxis=dict(
            categoryorder="array",
            categoryarray=shown_month_order,
            tickmode="array",
            tickvals=shown_month_order,
            tickangle=-30,
        ),
        margin=dict(b=120),
    )
    mean_std_fig.update_yaxes(tickformat=".0%")

    mean_std_file_name = "grounding_rate_mean_std_over_time"
    mean_std_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{mean_std_file_name}.html")
    mean_std_fig = with_paper_style(mean_std_fig, config=styler(18, 16))
    mean_std_fig.update_xaxes(
        tickangle=-30,
        tickmode="array",
        tickvals=shown_month_order,
    )
    mean_std_fig.write_image(
        f"{OUTPUT_PATH}/{CONF}/{mean_std_file_name}.pdf",
        format="pdf",
    )
    monthly_stats.to_csv(
        f"{OUTPUT_PATH}/{CONF}/{mean_std_file_name}_monthly_stats.csv",
        index=False,
    )

    plot_df[
        [
            "user_id",
            "conv_id",
            "turn_id",
            "time",
            "month",
            "num_cited_urls",
            "num_cited_external_urls",
            "num_cited_internal_urls",
            "grounding_rate",
            "unexplained_rate",
        ]
    ].to_csv(
        f"{OUTPUT_PATH}/{CONF}/{file_name}_samples.csv",
        index=False,
    )

    return plot_df


def plot_retrieved_url_counts_over_time_by_model():
    df = _prepare_source_count_df()
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
    file_name = "retrieved_url_counts_over_time_by_model"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 18), legend_pos=(0.8, 1.8))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def _plot_url_counts_grouped(df, group_col, file_name, xaxis_title):
    grouped = (
        df.groupby(group_col)[["num_retrieved_urls", "num_cited_urls"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    grouped = grouped.sort_values(("num_cited_urls", "mean"), ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped[group_col],
            y=grouped[("num_retrieved_urls", "mean")],
            name="Retrieved URLs",
            error_y=dict(
                type="data",
                array=grouped[("num_retrieved_urls", "sem")].fillna(0),
                visible=True,
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped[group_col],
            y=grouped[("num_cited_urls", "mean")],
            name="Cited URLs",
            error_y=dict(
                type="data",
                array=grouped[("num_cited_urls", "sem")].fillna(0),
                visible=True,
            ),
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis_title=xaxis_title,
        yaxis_title="Average # URLs per Turn",
        xaxis=dict(tickangle=-30),
    )
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 20))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def plot_url_counts_by_model():
    df = _prepare_source_count_df()
    df = df[df["model"].str.lower() != "unknown"].copy()
    _plot_url_counts_grouped(
        df,
        group_col="model",
        file_name="url_counts_by_model",
        xaxis_title="Model",
    )


def plot_url_counts_by_topic():
    df = _prepare_source_count_df()
    df = df[df["topic"].fillna("").str.lower() != "other"].copy()
    _plot_url_counts_grouped(
        df,
        group_col="topic",
        file_name="url_counts_by_topic",
        xaxis_title="Topic",
    )


def compare_safe_vs_retrieved_minus_safe_reachability():
    cache_path = f"{OUTPUT_PATH}/metadata/safe_vs_retrieved_minus_safe_reachability.csv"
    if os.path.exists(cache_path):
        comparison_df = pd.read_csv(cache_path)
    else:
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
        ).copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            retrieved = {
                item.get("url", "")
                for item in row["srcs_retrieved"]
                if isinstance(item, dict) and item.get("url", "")
            }
            safe = {
                item.get("url", "")
                for item in row["srcs_safe_urls"]
                if isinstance(item, dict) and item.get("url", "")
            }

            safe_reachable = count_reachable_urls(safe)
            retrieved_minus_safe = sorted(retrieved - safe)
            retrieved_minus_safe_reachable = count_reachable_urls(retrieved_minus_safe)

            rows.append(
                {
                    "month": row["month"],
                    "safe_total": len(safe),
                    "safe_reachable": safe_reachable,
                    "retrieved_minus_safe_total": len(retrieved_minus_safe),
                    "retrieved_minus_safe_reachable": retrieved_minus_safe_reachable,
                }
            )

        comparison_df = pd.DataFrame(rows)
        comparison_df.to_csv(cache_path, index=False)

    comparison_df["month"] = pd.to_datetime(comparison_df["month"], errors="coerce")
    monthly = (
        comparison_df.groupby("month")[
            [
                "safe_total",
                "safe_reachable",
                "retrieved_minus_safe_total",
                "retrieved_minus_safe_reachable",
            ]
        ]
        .sum()
        .sort_index()
    )
    if len(monthly) == 0:
        return comparison_df

    full_month_range = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS",
    )
    monthly = monthly.reindex(full_month_range, fill_value=0).rename_axis("month").reset_index()
    monthly["safe_reachability_rate"] = (
        monthly["safe_reachable"] / monthly["safe_total"].replace(0, pd.NA)
    ).fillna(0)
    monthly["retrieved_minus_safe_reachability_rate"] = (
        monthly["retrieved_minus_safe_reachable"]
        / monthly["retrieved_minus_safe_total"].replace(0, pd.NA)
    ).fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["safe_reachability_rate"],
            mode="lines+markers",
            name="Safe URLs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["retrieved_minus_safe_reachability_rate"],
            mode="lines+markers",
            name="Retrieved - Safe URLs",
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Reachability Rate",
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    file_name = "safe_vs_retrieved_minus_safe_reachability"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    return comparison_df


def plot_retrieved_safe_cited_positions(separate_cited_external_internal=False):
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
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
        file_name = "retrieved_safe_cited_positions"
        if separate_cited_external_internal:
            file_name += "_split_cited"
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 16))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

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

    file_name = "retrieved_safe_cited_positions_rank_violinplot"
    if separate_cited_external_internal:
        file_name += "_split_cited"
    violin_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    violin_fig = with_paper_style(violin_fig, config=styler(18, 16), legend_pos=None)
    violin_fig.update_xaxes(tickangle=0, tickfont=dict(size=16))
    violin_fig.update_layout(
        # height=680,
        # width=900,
        yaxis=dict(range=[main_rank_min, main_rank_max], autorange=False),
    )
    violin_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


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


def _load_first_existing_pickle(paths):
    for path in paths:
        if os.path.exists(path):
            return pd.read_pickle(path).copy()
    raise FileNotFoundError(f"None of these files exist: {paths}")


def _build_retrieved_safe_cited_rank_plot_rows(
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


def _plot_retrieved_safe_cited_ranks(plot_rows, file_name, yaxis_title):
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
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def plot_retrieved_safe_cited_tranco_ranks():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_tranco_ranks.pkl"
    ).copy()
    plot_rows = _build_retrieved_safe_cited_rank_plot_rows(
        df,
        retrieved_rank_col="ranks_srcs_retrieved",
        cited_rank_col="ranks_srcs_cited",
        min_rank=0,
    )
    _plot_retrieved_safe_cited_ranks(
        plot_rows,
        file_name="retrieved_safe_cited_tranco_ranks",
        yaxis_title="Tranco Rank",
    )


def plot_retrieved_safe_cited_judge_ranks():
    df = _load_first_existing_pickle(
        [
            f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.pkl",
            f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks_v2.pkl",
        ]
    )
    plot_rows = _build_retrieved_safe_cited_rank_plot_rows(
        df,
        retrieved_rank_col="reliability_scores_srcs_retrieved",
        cited_rank_col="reliability_scores_srcs_cited",
        rank_transform=lambda score: 5 - score,
    )
    _plot_retrieved_safe_cited_ranks(
        plot_rows,
        file_name="retrieved_safe_cited_judge_ranks",
        yaxis_title="Judge Rank (5 - Score)",
    )


def cited_sources_reachability():
    cache_path = f"{OUTPUT_PATH}/metadata/cited_sources_reachability.csv"
    if os.path.exists(cache_path):
        reachability_df = pd.read_csv(cache_path)
    else:
        df = pd.read_pickle(
            f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
        ).copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["month"] = df["time"].dt.to_period("M").dt.to_timestamp()

        unique_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            retrieved_urls = {
                item.get("url", "")
                for item in row["srcs_retrieved"]
                if isinstance(item, dict) and item.get("url", "")
            }
            retrieved_domains = {
                item.get("domain", "")
                for item in row["srcs_retrieved"]
                if isinstance(item, dict) and item.get("domain", "")
            }
            cited_items = [
                item
                for item in row["srcs_cited"]
                if isinstance(item, dict) and item.get("url", "")
            ]

            novel_cited_urls = sorted(
                {
                    item["url"]
                    for item in cited_items
                    if item["url"] not in retrieved_urls
                    and item.get("domain", "") not in retrieved_domains
                }
            )
            reachable_novel_cited = count_reachable_urls(novel_cited_urls)

            unique_rows.append(
                {
                    "month": row["month"],
                    "novel_cited_urls": novel_cited_urls,
                    "num_novel_cited_urls": len(novel_cited_urls),
                    "num_novel_cited_urls_hallucinated": len(novel_cited_urls) - reachable_novel_cited,
                }
            )

        reachability_df = pd.DataFrame(unique_rows)
        reachability_df.to_csv(cache_path, index=False)

    plot_hallucination_rate_over_time(
        reachability_df,
        total_col="num_novel_cited_urls",
        hallucinated_col="num_novel_cited_urls_hallucinated",
        file_name="novel_cited_url_hallucination_rate_over_time",
        yaxis_title="Hallucinated Cited URLs (%)",
    )


def plot_citations_round():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    ).copy()

    round_counts = {}
    for _, row in df.iterrows():
        cited_turns = [
            item.get("turn_index")
            for item in row["srcs_cited"]
            if isinstance(item, dict) and item.get("turn_index") is not None
        ]
        for cited_turn in cited_turns:
            round_counts[cited_turn] = round_counts.get(cited_turn, 0) + 1

    if not round_counts:
        return

    plot_df = (
        pd.DataFrame(
            {
                "round": list(round_counts.keys()),
                "count": list(round_counts.values()),
            }
        )
        .sort_values("round")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["round"],
            y=plot_df["count"],
            text=plot_df["count"],
            textposition="auto",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Citation Round",
        yaxis_title="Number of Citations",
    )
    file_name = "citation_round_distribution"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def check_url(url):
    if not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    session = requests.Session()
    session.headers.update(HEADERS)

    existing_statuses = {200, 201, 202, 204, 206, 301, 302, 303, 307, 308, 401, 403, 405, 406, 429}

    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
        )
        if response.status_code in existing_statuses:
            return True
    except requests.RequestException:
        pass

    try:
        response = session.get(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
            stream=True,
            headers={**HEADERS, "Range": "bytes=0-0"},
        )
        return response.status_code in existing_statuses
    except requests.RequestException:
        return False
    finally:
        session.close()
    

    # # Step 1: Format check
    # parsed = urlparse(url)
    # if parsed.scheme not in ("http", "https") or not parsed.netloc:
    #     return False

    # host = parsed.netloc

    # # Step 2: DNS Resolution
    # try:
    #     ip = socket.gethostbyname(host)
    # except Exception:
    #     return False

    # # Step 3: TCP Connection
    # port = 443 if parsed.scheme == "https" else 80
    # try:
    #     sock = socket.create_connection((host, port), timeout=TIMEOUT)
    # except Exception:
    #     return False

    # # Step 4: SSL Handshake (HTTPS only)
    # if parsed.scheme == "https":
    #     try:
    #         context = ssl.create_default_context()
    #         sock = context.wrap_socket(sock, server_hostname=host)
    #     except Exception:
    #         return False

    # # Step 5: HTTP Request (with browser headers)
    # try:
    #     response = requests.get(
    #         url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True
    #     )
    #     return True
    # except requests.exceptions.Timeout:
    #     return False
    # except requests.exceptions.RequestException:
    #     return False


def count_reachable_urls(urls):
    urls = sorted(set(urls))
    if not urls:
        return 0

    with ThreadPoolExecutor(max_workers=min(100, len(urls))) as executor:
        return sum(int(result) for result in executor.map(check_url, urls))


def plot_hallucination_rate_over_time(df, total_col, hallucinated_col, file_name, yaxis_title):
    plot_df = df.copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"], errors="coerce")
    plot_df = plot_df.dropna(subset=["month"])

    monthly = (
        plot_df.groupby("month")[[total_col, hallucinated_col]]
        .sum()
        .sort_index()
    )
    if len(monthly) == 0:
        return

    full_month_range = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS",
    )
    monthly = monthly.reindex(full_month_range, fill_value=0).rename_axis("month").reset_index()
    monthly["hallucination_rate"] = (
        monthly[hallucinated_col] / monthly[total_col].replace(0, pd.NA)
    )
    monthly["hallucination_rate"] = monthly["hallucination_rate"].fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["hallucination_rate"],
            mode="lines+markers",
            connectgaps=True,
            marker=dict(size=7),
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title=yaxis_title,
        xaxis=dict(
            tickmode="linear",
            dtick="M2",
            tickformat="%b %Y",
            tickangle=-30,
        ),
        margin=dict(b=90),
    )
    fig.update_yaxes(tickformat=".0%")
    fig.add_vline(
        x=pd.Timestamp("2024-11-01"),
        line_width=2,
        line_dash="dash",
        line_color="black",
    )
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 17))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def venn_diagram_of_sources():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    )

    retrieved_urls = [
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_retrieved"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    ]
    cited_urls = [
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_cited"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    ]
    print(len(retrieved_urls))
    print(len(cited_urls))
    cited_external = [x for x in cited_urls if x in retrieved_urls]
    print(len(cited_external))


    retrieved_urls = {
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_retrieved"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    }
    cited_urls = {
        _normalize_url_for_source_matching(src.get("url", ""))
        for sources in df["srcs_cited"]
        for src in sources
        if isinstance(src, dict) and src.get("url")
    }
    retrieved_set = {url for url in retrieved_urls if url}
    cited_set = {url for url in cited_urls if url}

    cited_external = cited_set & retrieved_set
    cited_internal = cited_set - retrieved_set

    print(len(retrieved_set))
    print(len(cited_set))
    print(len(cited_external))

    cited_external = 133264
    cited_internal = 17796
    external_valid = 127952
    external_non_valid = 5312
    internal_valid = 17198
    internal_non_valid = 598

    fig = go.Figure()

    outer_center_x = 2.0
    outer_center_y = 2.0
    outer_radius = 1.35
    inner_radius = 0.78

    def _disk_sector_points(cx, cy, radius, theta_start, theta_end, n_points=280):
        theta = np.linspace(theta_start, theta_end, n_points)
        x_arc = cx + radius * np.cos(theta)
        y_arc = cy + radius * np.sin(theta)
        return np.concatenate(([cx], x_arc, [cx])), np.concatenate(([cy], y_arc, [cy]))

    def _annulus_sector_points(
        cx,
        cy,
        radius_outer,
        radius_inner,
        theta_start,
        theta_end,
        n_points=280,
    ):
        theta_outer = np.linspace(theta_start, theta_end, n_points)
        theta_inner = np.linspace(theta_end, theta_start, n_points)

        x_outer = cx + radius_outer * np.cos(theta_outer)
        y_outer = cy + radius_outer * np.sin(theta_outer)
        x_inner = cx + radius_inner * np.cos(theta_inner)
        y_inner = cy + radius_inner * np.sin(theta_inner)
        return np.concatenate([x_outer, x_inner]), np.concatenate([y_outer, y_inner])

    inner_total = external_valid + external_non_valid
    ring_total = internal_valid + internal_non_valid
    inner_valid_fraction = external_valid / inner_total if inner_total > 0 else 0.5
    ring_valid_fraction = internal_valid / ring_total if ring_total > 0 else 0.5

    theta_start = np.pi / 2
    theta_full = theta_start + 2 * np.pi
    theta_inner_split = theta_start + 2 * np.pi * inner_valid_fraction
    theta_ring_split = theta_start + 2 * np.pi * ring_valid_fraction

    region_specs = [
        {
            "name": "Retrieved / External (Valid)",
            "region_type": "inner",
            "color": "#3765E5",
            "count": external_valid,
            "theta0": theta_start,
            "theta1": theta_inner_split,
        },
        {
            "name": "Retrieved / External (Non-valid)",
            "region_type": "inner",
            "color": "#9DB8FF",
            "count": external_non_valid,
            "theta0": theta_inner_split,
            "theta1": theta_full,
        },
        {
            "name": "Internal / Unexplained (Valid)",
            "region_type": "ring",
            "color": "#E45756",
            "count": internal_valid,
            "theta0": theta_start,
            "theta1": theta_ring_split,
        },
        {
            "name": "Internal / Unexplained (Non-valid)",
            "region_type": "ring",
            "color": "#FFB3A7",
            "count": internal_non_valid,
            "theta0": theta_ring_split,
            "theta1": theta_full,
        },
    ]

    count_annotations = []
    for spec in region_specs:
        if spec["region_type"] == "inner":
            x_points, y_points = _disk_sector_points(
                outer_center_x,
                outer_center_y,
                inner_radius,
                spec["theta0"],
                spec["theta1"],
            )
            label_radius = inner_radius * 0.58
        else:
            x_points, y_points = _annulus_sector_points(
                outer_center_x,
                outer_center_y,
                outer_radius,
                inner_radius,
                spec["theta0"],
                spec["theta1"],
            )
            label_radius = (outer_radius + inner_radius) / 2

        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=y_points,
                mode="lines",
                fill="toself",
                line=dict(color="rgba(0,0,0,0.35)", width=1.5),
                fillcolor=spec["color"],
                name=spec["name"],
                hovertemplate=(
                    f"{spec['name']}<br>Count: {spec['count']}<extra></extra>"
                ),
            )
        )
        mid_theta = 0.5 * (spec["theta0"] + spec["theta1"])
        count_annotations.append(
            (
                outer_center_x + label_radius * np.cos(mid_theta),
                outer_center_y + label_radius * np.sin(mid_theta),
                str(spec["count"]),
            )
        )

    fig.add_shape(
        type="circle",
        x0=outer_center_x - outer_radius,
        y0=outer_center_y - outer_radius,
        x1=outer_center_x + outer_radius,
        y1=outer_center_y + outer_radius,
        line=dict(color="rgba(0,0,0,0.7)", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.add_shape(
        type="circle",
        x0=outer_center_x - inner_radius,
        y0=outer_center_y - inner_radius,
        x1=outer_center_x + inner_radius,
        y1=outer_center_y + inner_radius,
        line=dict(color="rgba(0,0,0,0.7)", width=2),
        fillcolor="rgba(0,0,0,0)",
    )

    for x, y, text in count_annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=21, color="black"),
        )

    label_annotations = [
        (outer_center_x, outer_center_y + outer_radius + 0.18, "Cited URLs"),
    ]
    for x, y, text in label_annotations:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(size=16, color="black"),
        )

    fig.update_layout(
        xaxis=dict(visible=False, range=[0.58, 3.38]),
        yaxis=dict(visible=False, range=[0.58, 3.66], scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=26, pad=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    file_name = "venn_diagram_of_sources"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(
        fig,
        config=styler(18, 16),
        new_legend=dict(
            x=0.5,
            y=-0.03,
            xanchor="center",
            yanchor="top",
            orientation="h",
            entrywidthmode="fraction",
            entrywidth=0.48,
            traceorder="normal",
            bgcolor="rgba(255,255,255,0.75)",
            font=dict(size=14, color="black"),
        ),
    )
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def evaluate_source_tranco_ranks(
    separate_cited_external_internal=False,
    grounding_level="turn",
):
    grounding_level = _validated_grounding_level(grounding_level) 
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_tranco_ranks.pkl"
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
    box_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{violin_file_name}.html")
    box_fig = with_paper_style(box_fig, config=styler(26, 16), legend_pos=None)
    box_fig.update_xaxes(tickangle=0, tickfont=dict(size=26))
    box_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{violin_file_name}.pdf", format="pdf")

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
    paired_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{paired_file_name}.html")
    legend_pos = (0.98, 1.2) if separate_cited_external_internal else None
    paired_fig = with_paper_style(
        paired_fig,
        config=styler(18, 16),
        legend_pos=legend_pos,
    )
    paired_fig.update_layout(width=700 if separate_cited_external_internal else 500, height=400)
    paired_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{paired_file_name}.pdf", format="pdf")


def evaluate_source_topical_judge_ranks():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.pkl"
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

    def _avg_valid_score(scores):
        scores = _as_list(scores)
        valid_scores = []
        for score in scores:
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score_value):
                valid_scores.append(score_value)
        if not valid_scores:
            return np.nan
        return float(np.mean(valid_scores))

    source_score_specs = [
        ("retrieved_topic_reliability", "Retrieved", "reliability_scores_srcs_retrieved", "#636EFA"),
        ("cited_topic_reliability", "Cited", "reliability_scores_srcs_cited", "#EF553B"),
    ]
    for avg_col, _label, score_col, _color in source_score_specs:
        if score_col and score_col in df.columns:
            df[avg_col] = df[score_col].apply(_avg_valid_score)
        else:
            df[avg_col] = np.nan

    print("Average topical judge score ranges:")
    for avg_col, label, _score_col, _color in source_score_specs:
        subset = df[avg_col].dropna()
        if len(subset) == 0:
            print(f"{label}: no valid scores")
        else:
            print(f"{label}: {subset.min():.2f}-{subset.max():.2f}")

    score_df = df[[avg_col for avg_col, _label, _score_col, _color in source_score_specs]].copy()

    violin_fig = go.Figure()
    for avg_col, label, _score_col, color in source_score_specs:
        subset = score_df[avg_col].dropna()
        if len(subset) == 0:
            violin_fig.add_trace(
                go.Scatter(
                    x=[label],
                    y=[None],
                    mode="markers",
                    marker=dict(opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            continue
        violin_fig.add_trace(
            go.Violin(
                x=[label] * len(subset),
                y=subset,
                name=label,
                marker_color=color,
                line_color=color,
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            )
        )

    violin_fig.update_layout(
        xaxis_title="Source Type",
        yaxis_title="Average Score",
        xaxis=dict(
            categoryorder="array",
            categoryarray=["Retrieved", "Cited"],
        ),
        # height=600,
        # width=800,
    )
    violin_fig.update_yaxes(range=[1, 5])
    file_name = "source_topic_judge_rank_violinplot"
    violin_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    violin_fig = with_paper_style(violin_fig, config=styler(18, 16), legend_pos=None)
    # violin_fig.update_layout(height=600, width=800)
    violin_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    paired_fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=["Retrieved vs Cited"],
    )
    paired_specs = [
        ("retrieved_topic_reliability", "cited_topic_reliability", "Retrieved vs Cited", "#00CC96"),
    ]
    for idx, (x_col, y_col, label, color) in enumerate(paired_specs, start=1):
        subset = df[[x_col, y_col]].dropna()
        if len(subset) > 0:
            diagonal_min = min(subset[x_col].min(), subset[y_col].min())
            diagonal_max = max(subset[x_col].max(), subset[y_col].max())
            if diagonal_min == diagonal_max:
                diagonal_min -= 0.5
                diagonal_max += 0.5
            on_line = pd.Series(
                np.isclose(subset[y_col], subset[x_col]),
                index=subset.index,
            )
            above_line = subset[y_col] > subset[x_col]
            below_line = subset[y_col] < subset[x_col]
            above_count = int((above_line & ~on_line).sum())
            below_count = int((below_line & ~on_line).sum())
            on_line_count = int(on_line.sum())
            paired_fig.add_trace(
                go.Scatter(
                    x=subset[x_col],
                    y=subset[y_col],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=8),
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
            paired_fig.add_trace(
                go.Scatter(
                    x=[diagonal_min, diagonal_max],
                    y=[diagonal_min, diagonal_max],
                    mode="lines",
                    line=dict(color="black", dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
            paired_fig.update_xaxes(range=[diagonal_min, diagonal_max], row=1, col=idx)
            paired_fig.update_yaxes(range=[diagonal_min, diagonal_max], row=1, col=idx)
        else:
            above_count = 0
            below_count = 0
            on_line_count = 0
            paired_fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"x{'' if idx == 1 else idx} domain",
                yref=f"y{'' if idx == 1 else idx} domain",
                text="No paired data",
                showarrow=False,
                font=dict(size=14, color="gray"),
            )
            paired_fig.update_xaxes(range=[1, 5], row=1, col=idx)
            paired_fig.update_yaxes(range=[1, 5], row=1, col=idx)

        axis_suffix = "" if idx == 1 else str(idx)
        paired_fig.add_annotation(
            x=0.04,
            y=0.95,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"{above_count} points",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=13, color=color),
            bgcolor="rgba(255,255,255,0.78)",
        )
        paired_fig.add_annotation(
            x=0.54,
            y=0.54,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"{on_line_count} points",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=13, color="black"),
            bgcolor="rgba(255,255,255,0.78)",
        )
        paired_fig.add_annotation(
            x=0.96,
            y=0.05,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=f"{below_count} points",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=13, color=color),
            bgcolor="rgba(255,255,255,0.78)",
        )
        paired_fig.update_xaxes(title_text=x_col.replace("_", " ").title(), row=1, col=idx)
        paired_fig.update_yaxes(title_text=y_col.replace("_", " ").title(), row=1, col=idx)

    file_name = "source_topic_judge_rank_paired_plot"
    paired_fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    paired_fig = with_paper_style(paired_fig, config=styler(18, 16), legend_pos=None)
    paired_fig.update_layout(width=500, height=400)
    paired_fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")


def calculate_invivo_tranco_rank_correlation():
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_tranco_ranks.pkl"
    ).copy()

    tranco_col = "ranks_srcs_retrieved"
    invivo_col = "srcs_retrieved"
    missing_cols = [col for col in [tranco_col, invivo_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    id_cols = [col for col in ["user_id", "conv_id", "turn_id", "topic"] if col in df.columns]

    pair_rows = []
    sample_rows = []
    for sample_index, (_, row) in enumerate(df.iterrows()):
        row_id = {col: row[col] for col in id_cols}
        row_id["sample_index"] = int(sample_index)

        tranco_values = _as_list(row.get(tranco_col, []))
        invivo_items = _as_list(row.get(invivo_col, []))
        paired_count = min(len(tranco_values), len(invivo_items))
        if paired_count == 0:
            continue

        sample_tranco_ranks = []
        sample_invivo_ranks = []
        for idx in range(paired_count):
            tranco_rank = _as_valid_number(tranco_values[idx], min_value=1)
            if not np.isfinite(tranco_rank):
                continue

            invivo_item = invivo_items[idx]
            invivo_rank = np.nan
            if isinstance(invivo_item, dict):
                invivo_rank = _as_valid_number(invivo_item.get("ref_index"), min_value=0)
                if np.isfinite(invivo_rank):
                    invivo_rank += 1.0
            if not np.isfinite(invivo_rank):
                invivo_rank = float(idx + 1)

            sample_tranco_ranks.append(float(tranco_rank))
            sample_invivo_ranks.append(float(invivo_rank))
            pair_rows.append(
                {
                    **row_id,
                    "source_index": int(idx),
                    "tranco_rank": float(tranco_rank),
                    "invivo_rank": float(invivo_rank),
                }
            )

        if len(sample_tranco_ranks) == 0:
            continue
        sample_rows.append(
            {
                **row_id,
                "num_pairs": int(len(sample_tranco_ranks)),
                "avg_tranco_rank": float(np.mean(sample_tranco_ranks)),
                "avg_invivo_rank": float(np.mean(sample_invivo_ranks)),
            }
        )

    pair_columns = id_cols + [
        "sample_index",
        "source_index",
        "tranco_rank",
        "invivo_rank",
    ]
    pair_df = pd.DataFrame(pair_rows, columns=pair_columns)
    sample_columns = id_cols + [
        "sample_index",
        "num_pairs",
        "avg_tranco_rank",
        "avg_invivo_rank",
    ]
    sample_avg_df = pd.DataFrame(sample_rows, columns=sample_columns)
    if len(pair_df) == 0:
        raise ValueError("No valid Tranco/InVivo rank pairs found for plotting.")
    if len(sample_avg_df) == 0:
        raise ValueError("No valid per-sample averages found for plotting.")

    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    pair_df.to_csv(f"{output_dir}/invivo_tranco_rank_pairs.csv", index=False)
    sample_avg_df.to_csv(
        f"{output_dir}/invivo_tranco_rank_per_sample_avg.csv",
        index=False,
    )

    corr_pair_rows = []
    corr_sample_rows = []
    for sample_index, sample_df in pair_df.groupby("sample_index", sort=False):
        sample_meta = sample_df.iloc[0].to_dict()
        row_id = {col: sample_meta.get(col) for col in id_cols}
        row_id["sample_index"] = int(sample_index)

        # Keep only valid Tranco ranks (explicitly excluding -1).
        sample_valid = sample_df[pd.to_numeric(sample_df["tranco_rank"], errors="coerce") != -1].copy()
        sample_valid = sample_valid.dropna(subset=["tranco_rank", "invivo_rank"])

        if len(sample_valid) == 0:
            corr_sample_rows.append(
                {
                    **row_id,
                    "num_pairs_valid_tranco": 0,
                    "rank_corr_spearman": None,
                    "rank_corr_pearson": None,
                    "exact_order_match": None,
                }
            )
            continue

        sample_valid["tranco_order_rank"] = sample_valid["tranco_rank"].rank(
            method="first",
            ascending=True,
        )
        sample_valid["invivo_order_rank"] = sample_valid["invivo_rank"].rank(
            method="first",
            ascending=True,
        )

        for _, row in sample_valid.iterrows():
            corr_pair_rows.append(
                {
                    **row_id,
                    "source_index": int(row["source_index"]),
                    "tranco_rank": float(row["tranco_rank"]),
                    "invivo_rank": float(row["invivo_rank"]),
                    "tranco_order_rank": float(row["tranco_order_rank"]),
                    "invivo_order_rank": float(row["invivo_order_rank"]),
                }
            )

        if len(sample_valid) >= 2:
            rank_corr_spearman = sample_valid["tranco_order_rank"].corr(
                sample_valid["invivo_order_rank"],
                method="spearman",
            )
            rank_corr_pearson = sample_valid["tranco_order_rank"].corr(
                sample_valid["invivo_order_rank"],
                method="pearson",
            )
            tranco_order = tuple(
                sample_valid.sort_values(
                    ["tranco_order_rank", "source_index"],
                    ascending=[True, True],
                )["source_index"].tolist()
            )
            invivo_order = tuple(
                sample_valid.sort_values(
                    ["invivo_order_rank", "source_index"],
                    ascending=[True, True],
                )["source_index"].tolist()
            )
            exact_order_match = bool(tranco_order == invivo_order)
        else:
            rank_corr_spearman = None
            rank_corr_pearson = None
            exact_order_match = None

        corr_sample_rows.append(
            {
                **row_id,
                "num_pairs_valid_tranco": int(len(sample_valid)),
                "rank_corr_spearman": (
                    float(rank_corr_spearman) if pd.notna(rank_corr_spearman) else None
                ),
                "rank_corr_pearson": (
                    float(rank_corr_pearson) if pd.notna(rank_corr_pearson) else None
                ),
                "exact_order_match": exact_order_match,
            }
        )

    corr_pair_columns = id_cols + [
        "sample_index",
        "source_index",
        "tranco_rank",
        "invivo_rank",
        "tranco_order_rank",
        "invivo_order_rank",
    ]
    corr_sample_columns = id_cols + [
        "sample_index",
        "num_pairs_valid_tranco",
        "rank_corr_spearman",
        "rank_corr_pearson",
        "exact_order_match",
    ]
    corr_pair_df = pd.DataFrame(corr_pair_rows, columns=corr_pair_columns)
    corr_sample_df = pd.DataFrame(corr_sample_rows, columns=corr_sample_columns)
    corr_pair_df.to_csv(
        f"{output_dir}/invivo_tranco_rank_pairs_with_order_valid_tranco.csv",
        index=False,
    )
    corr_sample_df.to_csv(
        f"{output_dir}/invivo_tranco_rank_correlation_per_sample.csv",
        index=False,
    )

    corr_known = corr_sample_df.dropna(subset=["rank_corr_spearman"]) if len(corr_sample_df) > 0 else corr_sample_df
    exact_known = corr_sample_df.dropna(subset=["exact_order_match"]) if len(corr_sample_df) > 0 else corr_sample_df
    exact_order_match_rate = (
        float(pd.to_numeric(exact_known["exact_order_match"], errors="coerce").mean())
        if len(exact_known) > 0
        else None
    )
    corr_summary = {
        "method": (
            "Per-sample ranking correlation between Tranco and InVivo ranks "
            "over retrieved URLs with valid Tranco (Tranco != -1)"
        ),
        "n_pairs_valid_tranco": int(len(corr_pair_df)),
        "n_samples_total": int(len(corr_sample_df)),
        "n_samples_with_correlation": int(len(corr_known)),
        "mean_rank_corr_spearman": (
            float(pd.to_numeric(corr_known["rank_corr_spearman"], errors="coerce").mean())
            if len(corr_known) > 0
            else None
        ),
        "median_rank_corr_spearman": (
            float(pd.to_numeric(corr_known["rank_corr_spearman"], errors="coerce").median())
            if len(corr_known) > 0
            else None
        ),
        "mean_rank_corr_pearson": (
            float(pd.to_numeric(corr_known["rank_corr_pearson"], errors="coerce").mean())
            if len(corr_known) > 0
            else None
        ),
        "exact_order_match_rate": exact_order_match_rate,
    }
    to_json(corr_summary, f"{output_dir}/invivo_tranco_rank_correlation_summary.json")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_avg_df["avg_tranco_rank"],
            y=sample_avg_df["avg_invivo_rank"],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=6,
                opacity=0.45,
                color="#636EFA",
            ),
            hovertemplate=(
                "Avg Tranco Rank=%{x:.2f}<br>"
                "Avg InVivo Rank=%{y:.2f}<br>"
                "Pairs in sample=%{customdata}<extra></extra>"
            ),
            customdata=sample_avg_df["num_pairs"],
        )
    )
    fig.update_layout(
        xaxis_title="Average Tranco Rank (Per Sample)",
        yaxis_title="Average InVivo Rank (Per Sample)",
    )

    file_name = "invivo_tranco_rank_scatter"
    fig.write_html(f"{output_dir}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=None)
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

    summary = {
        "n_source_pairs": int(len(pair_df)),
        "n_samples": int(len(sample_avg_df)),
        "max_tranco_rank": float(pair_df["tranco_rank"].max()),
        "max_invivo_rank": float(pair_df["invivo_rank"].max()),
        "rank_correlation": corr_summary,
    }
    print(json.dumps(summary, indent=2))
    return sample_avg_df


def add_retrieved_safe_reliability_scores_to_topical_judge_ranks():
    df = _load_response_source_similarity_input()
    df_scores = pd.read_csv(
        f"{OUTPUT_PATH}/metadata/source_reliability_scores.csv"
    )

    required_score_cols = {"topic", "url", "score"}
    missing_score_cols = required_score_cols - set(df_scores.columns)
    if missing_score_cols:
        raise ValueError(
            f"Missing required score columns: {sorted(missing_score_cols)}"
        )

    source_specs = [
        ("srcs_retrieved", "reliability_scores_srcs_retrieved"),
        ("srcs_safe_urls", "reliability_scores_srcs_safe"),
        ("srcs_cited", "reliability_scores_srcs_cited"),
    ]
    missing_source_cols = [
        source_col
        for source_col, _score_col in source_specs
        if source_col not in df.columns
    ]
    if missing_source_cols:
        raise ValueError(
            f"Missing required source columns: {sorted(missing_source_cols)}"
        )

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

    def _normalize_topic(value):
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    def _normalize_url(value):
        if value is None or pd.isna(value):
            return ""
        return (
            str(value)
            .strip()
            .removesuffix("?utm_source=chatgpt.com")
            .removesuffix("&utm_source=chatgpt.com")
        )

    def _build_lookup(score_df):
        lookup = {}
        for _, score_row in score_df.iterrows():
            topic = _normalize_topic(score_row.get("topic", ""))
            url = _normalize_url(score_row.get("url", ""))
            if not topic or not url:
                continue
            try:
                score = float(score_row.get("score"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score):
                continue
            lookup[(topic, url)] = score
        return lookup

    score_lookup = _build_lookup(df_scores)

    def _score_sources(row, source_col):
        topic = _normalize_topic(row.get("topic", ""))
        scores = []
        for src in _as_list(row.get(source_col, [])):
            if isinstance(src, dict):
                url = _normalize_url(src.get("url", ""))
            else:
                url = _normalize_url(src)
            scores.append(score_lookup.get((topic, url), np.nan))
        return scores

    score_columns = {score_col: [] for _source_col, score_col in source_specs}
    for _, row in df.iterrows():
        for source_col, score_col in source_specs:
            score_columns[score_col].append(_score_sources(row, source_col))

    for score_col, scores in score_columns.items():
        df[score_col] = scores

    df.to_pickle(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.pkl"
    )
    df.to_csv(
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_topical_judge_ranks.csv",
        index=False,
    )
    print(f"Final length: {len(df)}")


def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")  # fallback
    

def count_token_density():
    max_token_count = 6000
    token_bin_size = 250
    selected_models = [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-5',
        'gpt-5-mini',
        # 'gpt-5-thinking',
        'gpt-5-2',
        # 'o3',
    ]
    df = pd.read_pickle(
        f"{OUTPUT_PATH}/metadata/retrieved_safe_cited_extracted_from_srcs.pkl"
    )
    df["model"] = df["openai_models"].apply(_primary_model)

    rows = []
    for _, row in df.iterrows():
        sources = row["srcs_retrieved"]
        model = row["model"]
        if model == "Unknown":
            continue

        token_count = 0
        for src in sources:
            if not isinstance(src, dict):
                continue
            text = f"{src.get('url', '')} {src.get('title', '')} {src.get('snippet', '')}"
            encoding = get_encoding(model)
            tokens = encoding.encode(text)
            token_count += len(tokens)

        rows.append({"model": model, "token_count": token_count})

    all_token_df = pd.DataFrame(rows)
    if all_token_df.empty:
        return

    fig_all = go.Figure()
    fig_all.add_trace(
        go.Histogram(
            x=all_token_df["token_count"],
            name="All Models",
            opacity=0.8,
            xbins=dict(start=0, end=max_token_count, size=token_bin_size),
        )
    )
    fig_all.update_layout(
        xaxis_title="Token Count per Retrieved Sources",
        yaxis_title="Frequency",
        margin=dict(b=80),
        xaxis=dict(range=[0, max_token_count]),
    )

    file_name = "token_density_all_models"
    fig_all.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig_all = with_paper_style(fig_all, config=styler(18, 16), legend_pos=None)
    fig_all.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    token_df = all_token_df[all_token_df["model"].isin(selected_models)].copy()
    if token_df.empty:
        return

    fig = go.Figure()
    for model in sorted(token_df["model"].unique()):
        model_counts = token_df[token_df["model"] == model]["token_count"]
        fig.add_trace(
            go.Histogram(
                x=model_counts,
                name=model,
                opacity=0.9,
                xbins=dict(start=0, end=max_token_count, size=token_bin_size),
            )
        )

    fig.update_layout(
        barmode="group",
        bargap=0.05,
        bargroupgap=0.0,
        xaxis_title="Token Count per Retrieved Sources",
        yaxis_title="Frequency",
        legend_title="Model",
        margin=dict(b=80),
        xaxis=dict(range=[0, max_token_count]),
    )

    file_name = "token_density_by_model"
    fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
    fig = with_paper_style(fig, config=styler(18, 16), legend_pos=(0.9, 1.2))
    fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    token_df.to_csv(
        f"{OUTPUT_PATH}/metadata/token_density_by_model.csv", index=False
    )
    all_token_df.to_csv(
        f"{OUTPUT_PATH}/metadata/token_density_all_models.csv", index=False
    )


if __name__ == "__main__":
    web_df = load_web_data_from_file(fmt="pkl")
    print(f"Loaded web data: {len(web_df)}")
    extract_retrieved_safe_cited_source(web_df)

    # count_unique_retrieved_safe_cited() needs
    # response_generation.extract_response_and_sources(web_df) run first --
    # it reads a different file than the line above produces. See
    # _prepare_source_count_df()'s docstring.
    # print(count_unique_retrieved_safe_cited())

    # plot_url_counts_over_time(separate_cited_external_internal=True)
    # plot_grounding_rate_violin_over_time()
    # plot_retrieved_url_counts_over_time_by_model()
    # plot_url_counts_by_model()
    # plot_url_counts_by_topic()

    # plot_retrieved_safe_cited_positions(separate_cited_external_internal=True)
    # plot_citations_round()

    # evaluate_source_tranco_ranks(separate_cited_external_internal=False, grounding_level="conversation")
    # evaluate_source_tranco_ranks(separate_cited_external_internal=True, grounding_level="conversation")
    # evaluate_source_topical_judge_ranks()

    # plot_top_domains(separate_cited_external_internal=True, grounding_level="conversation")
    # plot_top_domains(separate_cited_external_internal=False, grounding_level="conversation")
    # plot_top_domains_by_selected_topics(separate_cited_external_internal=True)
    # plot_top_domains_by_model(separate_cited_external_internal=True)

    # venn_diagram_of_sources()

    # compute_average_citations_and_retrievals_per_response(grounding_level="conversation")
    # compute_average_citations_and_retrievals_per_response_by_openai_model(
    #     grounding_level="conversation"
    # )
    # plot_retrieved_urls_per_web_query_histogram()

    # _user_prompt_domain_counter()

    # evaluate_unique_retrieved_domains_by_platform()

    # plot_chatgpt_claude_retrieved_domain_rank_scatter()
    pass
