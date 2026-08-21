import ast
import asyncio
import csv
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.common_io import OUTPUT_PATH
from openai import OpenAI
from tqdm import tqdm
from src.prompts.evaluator_prompts import (
    SYSTEM_PROMPT_CHARAC,
    SYSTEM_PROMPT_ENTITY_SPECIFICITY,
    SYSTEM_PROMPT_GEOGRAPHIC_SPECIFICITY,
    SYSTEM_PROMPT_NUMERIC_SPECIFICITY,
    SYSTEM_PROMPT_QUERY_REASON,
    SYSTEM_PROMPT_QUERY_REASON_VALIDATOR,
    SYSTEM_PROMPT_TEMPORAL_SPECIFICITY,
    USER_PROMPT_CHARAC,
    USER_PROMPT_ENTITY_SPECIFICITY,
    USER_PROMPT_GEOGRAPHIC_SPECIFICITY,
    USER_PROMPT_NUMERIC_SPECIFICITY,
    USER_PROMPT_QUERY_REASON,
    USER_PROMPT_QUERY_REASON_VALIDATOR,
    USER_PROMPT_TEMPORAL_SPECIFICITY,
)

load_dotenv()

DEFAULT_MODELS = [
    "gpt-5.3-chat-latest",
    "claude-sonnet-4-6",
    "grok-4.3",
    "deepseek-v4-flash",
]
OPENAI_REPLAY_MODELS = [
    "o4-mini-2025-04-16",
    "gpt-4.1-mini-2025-04-14",
    "gpt-5.3-chat-latest",
]
INPUT_DIR = Path(f"{OUTPUT_PATH}/replays")
OUTPUT_DIR = Path(f"{OUTPUT_PATH}/replays/extracted")
PLOT_OUTPUT_DIR = Path(f"{OUTPUT_PATH}/replays/plots")
WEB_CALLS_CHARACTERIZATION_PATH = Path(
    f"{OUTPUT_PATH}/metadata/web_calls_characterization.csv"
)
REPLAY_SAMPLE_CHARACTERIZATION_PATH = Path(
    f"{OUTPUT_PATH}/metadata/replay_samples_web_calls_characterization.csv"
)
REPLAY_SAMPLE_SOURCE_PATH = INPUT_DIR / "gpt-5.3-chat-latest.json"
REPLAY_QUERY_EVAL_OUTPUT_DIR = OUTPUT_DIR / "query_reformulations"
REPLAY_URLS_CONTENT_PATH = OUTPUT_DIR / "replay_response_and_sources_url_content.json"
REPLAY_CLAIM_CACHE_PATH = OUTPUT_DIR / "replay_response_source_claim_chunks_cache.json"

PRIMARY_TRIGGER_LABEL_MAP = {
    "High-Investment Recommendation": "High-Investment",
    "Volatile/Temporal Information": "Temporal Information",
    "Low Confidence/Niche Fact": "Low Confidence Fact",
    "Unfamiliar Term/Typo": "Unfamiliar Term",
    "High-Stakes Accuracy": "High-Stakes Accuracy",
    "External Reference": "External Reference",
    "User Verification": "User Verification",
    "Attribution/Sourcing Needed": "Attribution Needed",
    "Explicit Command": "Explicit Command",
}
EXCLUDED_PRIMARY_TRIGGERS = {"None of the Above", "OpenAI Product Info", ""}
NO_CHARACTERIZATION_LABEL = "No Characterization"


def _infer_provider(model_name, row):
    provider = row.get("replay_provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip().lower()

    model_key = (model_name or "").strip().lower()
    if model_key.startswith("claude"):
        return "claude"
    if model_key.startswith("grok"):
        return "grok"
    if model_key.startswith("deepseek"):
        return "deepseek"
    return "openai"


def _parse_followed_web_policy(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    text = value.strip()
    if not text:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = text[start : end + 1]
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue

    return {}


def _parse_eval_json(text):
    if not isinstance(text, str):
        return {}

    stripped = text.strip()
    if not stripped:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = stripped[start : end + 1]
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue

    return {}


def _payload_for_row(row):
    for key, value in row.items():
        if isinstance(value, dict) and "response" in value and "output_text" in value:
            return key, value
    return None, None


def _queries_from_openai_response(response):
    queries = []
    sources = []

    for item in response.get("output", []) or []:
        if item.get("type") != "web_search_call":
            continue

        action = item.get("action") or {}
        call_queries = []
        call_sources = []

        action_queries = action.get("queries")
        if isinstance(action_queries, list):
            for query in action_queries:
                if isinstance(query, str) and query.strip():
                    call_queries.append(query.strip())

        query = action.get("query") or item.get("query")
        if isinstance(query, str) and query.strip():
            if query.strip() not in call_queries:
                call_queries.append(query.strip())

        for source in action.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            call_sources.append(
                {
                    "url": source.get("url"),
                    "title": source.get("title"),
                    "snippet": source.get("snippet"),
                }
            )
        queries.append(call_queries)
        sources.append(call_sources)

    return queries, sources


def _cited_sources_from_openai_response(response):
    cited_sources = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            for annotation in content.get("annotations", []) or []:
                if not isinstance(annotation, dict):
                    continue
                cited_sources.append(
                    {
                        "url": annotation.get("url"),
                        "title": annotation.get("title"),
                        "text": annotation.get("text"),
                        "start_index": annotation.get("start_index"),
                        "end_index": annotation.get("end_index"),
                    }
                )
    return cited_sources


def _queries_from_anthropic_response(response):
    queries = []
    sources = []
    pending_call_queries = []

    for block in response.get("content", []) or []:
        block_type = block.get("type")
        if block_type in {"server_tool_use", "tool_use"} and block.get("name") == "web_search":
            tool_input = block.get("input") or {}
            call_queries = []

            multi_queries = tool_input.get("queries")
            if isinstance(multi_queries, list):
                for query in multi_queries:
                    if isinstance(query, str) and query.strip():
                        call_queries.append(query.strip())

            query = tool_input.get("query")
            if isinstance(query, str) and query.strip():
                if query.strip() not in call_queries:
                    call_queries.append(query.strip())

            queries.append(call_queries)
            pending_call_queries.append(call_queries)

        if block_type == "web_search_tool_result":
            content = block.get("content")
            if not isinstance(content, list):
                content = [content] if content else []

            call_sources = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                call_sources.append(
                    {
                        "url": item.get("url"),
                        "title": item.get("title"),
                        "snippet": item.get("text") or item.get("snippet"),
                    }
                )
            sources.append(call_sources)
            if pending_call_queries:
                pending_call_queries.pop(0)
            else:
                queries.append([])

    return queries, sources


def _cited_sources_from_anthropic_response(response):
    cited_sources = []
    for block in response.get("content", []) or []:
        if block.get("type") != "text":
            continue
        response_text = block.get("text")
        for citation in block.get("citations", []) or []:
            if not isinstance(citation, dict):
                continue
            cited_sources.append(
                {
                    "url": citation.get("url"),
                    "title": citation.get("title") or citation.get("document_title"),
                    "text": citation.get("cited_text"),
                    "response_text": response_text,
                    "start_index": citation.get("start_index"),
                    "end_index": citation.get("end_index"),
                }
            )
    return cited_sources


def _response_text_blocks_from_anthropic_response(response):
    blocks_out = []
    for block in response.get("content", []) or []:
        if block.get("type") != "text":
            continue
        text = str(block.get("text", "") or "")
        citations = []
        for citation in block.get("citations", []) or []:
            if not isinstance(citation, dict):
                continue
            url = _clean_citation_url(citation.get("url"))
            if url:
                citations.append(
                    {
                        "url": url,
                        "title": citation.get("title") or citation.get("document_title"),
                        "cited_text": citation.get("cited_text"),
                        "start_index": citation.get("start_index"),
                        "end_index": citation.get("end_index"),
                    }
                )
        blocks_out.append(
            {
                "text": text,
                "citations": citations,
            }
        )
    return blocks_out


def _extract_web_artifacts(provider, response):
    provider = (provider or "").lower()
    if provider in {"openai", "grok"}:
        return _queries_from_openai_response(response)
    if provider in {"claude", "deepseek"}:
        return _queries_from_anthropic_response(response)
    return [], []


def _has_web_tool_call(provider, response):
    provider = (provider or "").lower()
    if provider in {"openai", "grok"}:
        return any(
            isinstance(item, dict) and item.get("type") == "web_search_call"
            for item in (response.get("output", []) or [])
        )
    if provider in {"claude", "deepseek"}:
        return any(
            isinstance(block, dict)
            and (
                (
                    block.get("type") in {"server_tool_use", "tool_use"}
                    and block.get("name") == "web_search"
                )
                or block.get("type") == "web_search_tool_result"
            )
            for block in (response.get("content", []) or [])
        )
    return False


def _extract_cited_sources(provider, response):
    provider = (provider or "").lower()
    if provider in {"openai", "grok"}:
        return _cited_sources_from_openai_response(response)
    if provider in {"claude", "deepseek"}:
        return _cited_sources_from_anthropic_response(response)
    return []


def _clean_sources(sources):
    cleaned = []
    seen = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        normalized = {
            "url": source.get("url"),
            "title": source.get("title"),
            "snippet": source.get("snippet"),
        }
        key = (
            normalized["url"],
            normalized["title"],
            normalized["snippet"],
        )
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def _clean_nested_sources(nested_sources):
    return [_clean_sources(source_list) for source_list in nested_sources]


def _count_nested_sources(nested_sources):
    return sum(len(source_list) for source_list in nested_sources)


def _drop_empty_query_groups(web_queries, sources_retrieved):
    cleaned_queries = []
    cleaned_sources = []
    max_len = max(len(web_queries), len(sources_retrieved))

    for idx in range(max_len):
        query_group = web_queries[idx] if idx < len(web_queries) else []
        source_group = sources_retrieved[idx] if idx < len(sources_retrieved) else []
        if query_group:
            cleaned_queries.append(query_group)
            cleaned_sources.append(source_group)

    return cleaned_queries, cleaned_sources


def _clean_citation_url(url):
    if not isinstance(url, str):
        return ""
    return url.strip()


def _clean_cited_sources(sources):
    cleaned = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        normalized = {
            "url": _clean_citation_url(source.get("url")),
            "title": source.get("title"),
            "text": source.get("text"),
            "response_text": source.get("response_text"),
            "start_index": source.get("start_index"),
            "end_index": source.get("end_index"),
        }
        cleaned.append(
            {k: v for k, v in normalized.items() if v not in (None, "", [], {})}
        )
    return cleaned


def _clean_response_text_blocks(blocks):
    cleaned = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        citations = []
        for citation in block.get("citations", []) or []:
            if not isinstance(citation, dict):
                continue
            normalized_citation = {
                "url": _clean_citation_url(citation.get("url")),
                "title": citation.get("title"),
                "cited_text": citation.get("cited_text"),
                "start_index": citation.get("start_index"),
                "end_index": citation.get("end_index"),
            }
            citations.append(
                {k: v for k, v in normalized_citation.items() if v not in (None, "", [], {})}
            )
        cleaned.append(
            {
                "text": str(block.get("text", "") or ""),
                "citations": citations,
            }
        )
    return cleaned


def _count_unique_cited_sources(sources):
    unique = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        unique.add((source.get("url"), source.get("title")))
    return len(unique)


def _source_item_count(items, key="url", unique=True):
    flat_items = _flatten_source_items(items)
    if unique:
        values = {
            item.get(key, "")
            for item in flat_items
            if isinstance(item, dict) and (key is None or item.get(key, ""))
        }
        return len(values)
    return sum(
        1
        for item in flat_items
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


def _load_replay_json(path):
    with open(path) as f:
        raw_text = f.read()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        try:
            parsed = ast.literal_eval(raw_text)
        except (ValueError, SyntaxError) as fallback_exc:
            raise ValueError(
                f"Failed to parse replay file `{path}` as JSON or Python literal. "
                f"JSON error: {exc}"
            ) from fallback_exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Replay file `{path}` did not parse to a dict; got {type(parsed).__name__}."
        )
    return parsed


def extract_model_file(model_name, input_dir, tool_choice=None):
    path = input_dir / f"{model_name}.json"
    replay_data = _load_replay_json(path)

    rows = []
    for result_key, row in replay_data.items():
        if row.get("skipped_replay"):
            continue

        selected_tool_choice = tool_choice
        if tool_choice is None:
            selected_tool_choice, payload = _payload_for_row(row)
        else:
            payload = row.get(tool_choice)
        if payload is None:
            continue

        response = payload.get("response") or {}
        provider = _infer_provider(model_name, row)
        has_web_tool_call = _has_web_tool_call(provider, response)
        web_queries, sources_retrieved = _extract_web_artifacts(provider, response)
        sources_cited = _extract_cited_sources(provider, response)
        response_text_blocks = (
            _response_text_blocks_from_anthropic_response(response)
            if provider in {"claude", "deepseek"}
            else []
        )
        sources_retrieved = _clean_nested_sources(sources_retrieved)
        web_queries, sources_retrieved = _drop_empty_query_groups(
            web_queries, sources_retrieved
        )
        sources_cited = _clean_cited_sources(sources_cited)
        response_text_blocks = _clean_response_text_blocks(response_text_blocks)

        rows.append(
            {
                "model": model_name,
                "provider": provider,
                "tool_choice": tool_choice,
                "result_key": result_key,
                "sample_idx": row.get("sample_idx"),
                "sample_source": row.get("sample_source"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "user_prompt": row.get("user_prompt"),
                "final_response": payload.get("output_text", ""),
                "has_web_tool_call": has_web_tool_call,
                "web_queries": web_queries,
                "sources_retrieved": sources_retrieved,
                "sources_cited": sources_cited,
                "response_text_blocks": response_text_blocks,
                "num_web_search_calls": len(web_queries),
                "num_sources_retrieved": _count_nested_sources(sources_retrieved),
                "num_sources_cited": len(sources_cited),
                "num_unique_sources_cited": _count_unique_cited_sources(sources_cited),
                "error": payload.get("error"),
            }
        )

    return rows


def _sample_key(row):
    return row.get("result_key")


def _single_model_json_payload(rows):
    payload = {}
    for row in rows:
        sample_id = _sample_key(row)
        payload[sample_id] = {
            key: value
            for key, value in row.items()
            if key != "result_key"
        }
    return payload


def _all_models_json_payload(rows):
    payload = {}
    for row in rows:
        sample_id = _sample_key(row)
        if sample_id not in payload:
            payload[sample_id] = {
                "sample_idx": row.get("sample_idx"),
                "sample_source": row.get("sample_source"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "user_prompt": row.get("user_prompt"),
                "models": {},
            }
        payload[sample_id]["models"][row.get("model")] = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "model",
                "result_key",
                "sample_idx",
                "sample_source",
                "conv_id",
                "turn_id",
                "user_prompt",
            }
        }
    return payload


def _json_payload(rows):
    models = {row.get("model") for row in rows}
    if len(models) <= 1:
        return _single_model_json_payload(rows)
    return _all_models_json_payload(rows)


def save_outputs(rows, output_prefix):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{output_prefix}.json", "w") as f:
        json.dump(_json_payload(rows), f, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(f"{output_prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_pickle(f"{output_prefix}.pkl")
        return df
    except ModuleNotFoundError:
        return rows


def save_replay_retrieved_domain_counts_per_model(
    input_path=OUTPUT_DIR / "all_models.json",
    output_dir=OUTPUT_DIR,
):
    import pandas as pd
    from collections import Counter

    payload = _load_replay_json(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_domain_counts = {}
    for sample_payload in payload.values():
        if not isinstance(sample_payload, dict):
            continue
        model_payloads = sample_payload.get("models", {})
        if not isinstance(model_payloads, dict):
            continue

        for model_name, model_payload in model_payloads.items():
            if not isinstance(model_payload, dict):
                continue
            if model_name not in model_domain_counts:
                model_domain_counts[model_name] = Counter()

            for source_group in model_payload.get("sources_retrieved", []) or []:
                if not isinstance(source_group, list):
                    continue
                for source in source_group:
                    if not isinstance(source, dict):
                        continue
                    domain = _normalize_domain_for_top_plots(
                        source.get("domain")
                        or urlparse(str(source.get("url", "") or "")).netloc
                    )
                    if domain:
                        model_domain_counts[model_name][domain] += 1

    saved_files = {}
    for model_name, counts in model_domain_counts.items():
        df = pd.DataFrame(
            [
                {"domain": domain, "count": int(count)}
                for domain, count in counts.most_common()
            ]
        )
        model_slug = str(model_name).replace(".", "-")
        output_path = output_dir / f"retrieved_domain_counts__{model_slug}.csv"
        df.to_csv(output_path, index=False)
        saved_files[model_name] = str(output_path)
        print(
            f"Saved {len(df)} retrieved-domain rows for "
            f"{_model_label(model_name)} to {output_path}"
        )

    return saved_files


def save_sample_web_queries(rows, output_prefix):
    query_rows = []
    for row in rows:
        query_rows.append(
            {
                "model": row.get("model"),
                "provider": row.get("provider"),
                "result_key": row.get("result_key"),
                "sample_idx": row.get("sample_idx"),
                "sample_source": row.get("sample_source"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "web_queries": row.get("web_queries", []),
            }
        )
    save_outputs(query_rows, output_prefix)


def save_prompt_and_model_web_queries(rows, output_path):
    grouped = {}
    for row in rows:
        result_key = row.get("result_key")
        if result_key not in grouped:
            grouped[result_key] = {
                "user_prompt": row.get("user_prompt"),
            }
        grouped[result_key][row.get("model")] = row.get("web_queries", [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False)


def _extract_rows_for_models(model_names, input_dir=INPUT_DIR, tool_choice=None):
    all_rows = []
    for model_name in model_names:
        rows = extract_model_file(model_name, input_dir, tool_choice=tool_choice)
        all_rows.extend(rows)
    return all_rows


def _common_result_keys_for_models(rows, model_names):
    key_sets = []
    for model_name in model_names:
        model_keys = {
            row.get("result_key")
            for row in rows
            if row.get("model") == model_name and row.get("result_key")
        }
        if model_keys:
            key_sets.append(model_keys)

    if not key_sets:
        return set()
    return set.intersection(*key_sets)


def _filter_rows_to_common_samples(rows, model_names):
    common_keys = _common_result_keys_for_models(rows, model_names)
    if not common_keys:
        return []
    return [row for row in rows if row.get("result_key") in common_keys]


def _filter_rows_to_samples_with_web_calls_for_all_models(rows, model_names):
    grouped_calls = {}
    for row in rows:
        result_key = row.get("result_key")
        model_name = row.get("model")
        if not result_key or model_name not in model_names:
            continue
        grouped_calls.setdefault(result_key, {})[model_name] = bool(
            row.get("has_web_tool_call", False)
        )

    valid_keys = {
        result_key
        for result_key, model_calls in grouped_calls.items()
        if all(model_calls.get(model_name, False) for model_name in model_names)
    }
    if not valid_keys:
        return []
    return [row for row in rows if row.get("result_key") in valid_keys]


def _normalize_topic_name(topic):
    topic = str(topic or "").strip()
    if topic == "GPT":
        return "OpenAI Product Info"
    return topic


def _build_topic_lookup():
    from src.web_search_decision.chatgpt_extraction import load_web_data_from_file, load_whole_data_from_file

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return {}

    whole_df = load_whole_data_from_file(fmt="pkl").copy()
    web_df = load_web_data_from_file(fmt="pkl").copy()
    df = pd.concat([whole_df, web_df], ignore_index=True)
    df["conv_id"] = df["conv_id"].astype(str)
    df["turn_id"] = df["turn_id"].astype(str)
    df["topic"] = df["topic"].fillna("").apply(_normalize_topic_name)
    df = df.dropna(subset=["conv_id", "turn_id"])
    df = df.drop_duplicates(subset=["conv_id", "turn_id"], keep="last")
    return {
        (row["conv_id"], row["turn_id"]): row["topic"]
        for _, row in df.iterrows()
        if str(row["topic"]).strip()
    }


def _load_primary_triggers(
    metadata_path=WEB_CALLS_CHARACTERIZATION_PATH,
    judge_model=None,
):
    trigger_by_turn = {}
    trigger_counts = {}

    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if judge_model is not None:
                row_judge_model = str(row.get("judge_model", "") or "").strip()
                if row_judge_model != str(judge_model):
                    continue
            conv_id = str(row.get("conv_id", "") or "").strip()
            turn_id = str(row.get("turn_id", "") or "").strip()
            if not conv_id or not turn_id:
                continue

            policy = _parse_followed_web_policy(row.get("followed_web_policy"))
            primary_trigger = str(policy.get("primary_trigger", "") or "").strip()
            if primary_trigger in EXCLUDED_PRIMARY_TRIGGERS:
                continue

            trigger_label = PRIMARY_TRIGGER_LABEL_MAP.get(
                primary_trigger,
                primary_trigger,
            )
            trigger_by_turn[(conv_id, turn_id)] = trigger_label
            trigger_counts[trigger_label] = trigger_counts.get(trigger_label, 0) + 1

    ordered_triggers = sorted(
        trigger_counts,
        key=lambda label: (-trigger_counts[label], label),
    )
    return trigger_by_turn, ordered_triggers


def _group_rows_by_sample(rows):
    grouped = {}
    for row in rows:
        result_key = row.get("result_key")
        if result_key not in grouped:
            grouped[result_key] = {
                "result_key": result_key,
                "sample_idx": row.get("sample_idx"),
                "sample_source": row.get("sample_source"),
                "conv_id": str(row.get("conv_id", "") or ""),
                "turn_id": str(row.get("turn_id", "") or ""),
                "user_prompt": row.get("user_prompt"),
                "models": {},
            }
        grouped[result_key]["models"][row.get("model")] = bool(row.get("web_queries", []))
    return grouped


def _run_judge(client, model_name, system_prompt, user_prompt):
    provider = _infer_provider(model_name, {"replay_provider": None})
    if provider in {"openai", "grok"}:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response.output_text
    else:
        response = client.messages.create(
            model=model_name,
            max_tokens=2048,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = "\n".join(
            block.text
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", "") == "text" and getattr(block, "text", "")
        )
    return {
        "raw_judgment": raw_text,
        "parsed_judgment": _parse_eval_json(raw_text),
    }


def _safe_float(value):
    try:
        if value in (None, "", "nan", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_skipped_sample_idx(sample_idx):
    from src.replays.chat_replayer import SKIPPED_REPLAY_SAMPLE_INDICES

    try:
        return int(sample_idx) in SKIPPED_REPLAY_SAMPLE_INDICES
    except (TypeError, ValueError):
        return False


def _mean_or_na(values):
    if not values:
        return "NA"
    return f"{(sum(values) / len(values)):.2f}"


def _mean_with_bootstrap_ci_or_na(values, confidence=0.95):
    if not values:
        return "NA"
    ci = _bootstrap_mean_ci(values, confidence=confidence)
    mean = ci.get("mean")
    ci_low = ci.get("ci_low")
    ci_high = ci.get("ci_high")
    if any(not np.isfinite(value) for value in [mean, ci_low, ci_high]):
        return "NA"
    confidence_pct = int(round(confidence * 100))
    return f"{mean:.2f} [{confidence_pct}% CI: {ci_low:.2f}, {ci_high:.2f}]"


def _safe_json_value(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(value)
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
        return default
    return value


def _dedupe_preserve_order(values):
    seen = set()
    deduped = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_reason_transition_endpoint(value):
    if value is None:
        return ""
    endpoint = str(value).strip()
    endpoint = endpoint.strip("()").replace(" ", "")
    if not endpoint:
        return ""
    lowered = endpoint.lower()
    if lowered in {
        "u",
        "user",
        "userquery",
        "user_query",
        "userprompt",
        "user_prompt",
    }:
        return "U"
    return endpoint


def _normalize_query_reason_label(value):
    label = str(value or "").strip()
    lowered = label.lower()
    if not lowered:
        return ""
    if "other" in lowered:
        return "Other"
    if "hybrid" in lowered:
        return "Hybrid"
    if "expansion" in lowered:
        return "Query Expansion"
    if "rewriting" in lowered or "rewrite" in lowered:
        return "Query Rewriting"
    return label


def _save_replay_query_eval_records(records, output_stem):
    output_dir = REPLAY_QUERY_EVAL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{output_stem}.json", "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    try:
        import pandas as pd

        df = pd.DataFrame(records)
        df.to_csv(output_dir / f"{output_stem}.csv", index=False)
        df.to_pickle(output_dir / f"{output_stem}.pkl")
        return df
    except ModuleNotFoundError:
        fieldnames = list(records[0].keys()) if records else []
        with open(output_dir / f"{output_stem}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    }
                )
        return records


def _load_replay_query_eval_df(output_stem):
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None

    pkl_path = REPLAY_QUERY_EVAL_OUTPUT_DIR / f"{output_stem}.pkl"
    csv_path = REPLAY_QUERY_EVAL_OUTPUT_DIR / f"{output_stem}.csv"
    if pkl_path.exists():
        return pd.read_pickle(pkl_path).copy()
    if csv_path.exists():
        return pd.read_csv(csv_path).copy()
    return None


def _build_replay_query_eval_rows(model_names=DEFAULT_MODELS):
    rows = _extract_rows_for_models(model_names)
    rows = _filter_rows_to_common_samples(rows, model_names)
    replay_rows = []
    for row in rows:
        replay_rows.append(
            {
                "model": row.get("model"),
                "provider": row.get("provider"),
                "result_key": row.get("result_key"),
                "sample_idx": row.get("sample_idx"),
                "sample_source": row.get("sample_source"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "user_prompt": row.get("user_prompt", ""),
                "user_msg_history": [row.get("user_prompt", "")]
                if row.get("user_prompt")
                else [],
                "assistant_msg_history": [],
                "web_queries": row.get("web_queries", []),
                "thoughts_list": [[] for _ in row.get("web_queries", [])],
                "sources": row.get("sources_retrieved", []),
            }
        )
    return replay_rows


def _model_label(model_name):
    replacements = {
        "gpt-5.3-chat-latest": "GPT-5.3",
        "gpt-4.1-2025-04-14": "GPT-4.1",
        "gpt-4.1-mini-2025-04-14": "GPT-4.1 Mini",
        "o3-2025-04-16": "o3",
        "o4-mini-2025-04-16": "o4-mini",
        "grok-4.3": "Grok-4.3",
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
        "deepseek-v4-flash": "DeepSeek V4 Flash",
    }
    return replacements.get(str(model_name), str(model_name))


def _outcome_key(base_called, model_called):
    if base_called and model_called:
        return "base_called_and_model_called"
    if base_called and not model_called:
        return "base_called_and_model_not_called"
    if not base_called and not model_called:
        return "base_not_called_and_model_not_called"
    return "base_not_called_and_model_called"


def _clean_web_query_groups(value):
    if not isinstance(value, list):
        return []

    cleaned_groups = []
    for query_group in value:
        if not isinstance(query_group, list):
            continue
        cleaned_group = [
            str(query).strip()
            for query in query_group
            if isinstance(query, str) and query.strip()
        ]
        if cleaned_group:
            cleaned_groups.append(cleaned_group)
    return cleaned_groups


def _model_subset_slug(model_names):
    if isinstance(model_names, str):
        model_names = [model_names]
    labels = [str(model_name).replace(".", "-") for model_name in model_names]
    return "__".join(labels)


def _count_terms_local(text, remove_stopwords=False):
    text = "" if text is None else str(text).strip()
    if not text:
        return 0
    if remove_stopwords:
        from src.query_formulation.query_reformulations import preprocess_text_in_chunks

        return len(preprocess_text_in_chunks(text))
    return len(text.split())


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


def _flatten_source_items(items):
    flattened = []
    if not isinstance(items, list):
        return flattened
    for item in items:
        if isinstance(item, dict):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(
                nested_item
                for nested_item in item
                if isinstance(nested_item, dict)
            )
    return flattened


def _domain_counter_from_rows(rows, col_name, top_k=10):
    counts = {}
    for row in rows:
        items = _flatten_source_items(row.get(col_name, []))
        for item in items:
            domain = _normalize_domain_for_top_plots(
                item.get("domain") or urlparse(str(item.get("url", ""))).netloc
            )
            if domain:
                counts[domain] = counts.get(domain, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def _cited_domain_counter_split_from_rows(rows, top_k=10):
    retrieved_cited_counts = {}
    parametric_cited_counts = {}

    for row in rows:
        retrieved_items = _flatten_source_items(row.get("sources_retrieved", []))
        cited_items = _flatten_source_items(row.get("sources_cited", []))
        if not cited_items:
            continue

        retrieved_urls = {
            _normalize_url_for_source_matching(item.get("url", ""))
            for item in retrieved_items
            if isinstance(item, dict) and item.get("url", "")
        }

        for item in cited_items:
            if not isinstance(item, dict):
                continue
            cited_url = _normalize_url_for_source_matching(item.get("url", ""))
            if not cited_url:
                continue
            domain = _normalize_domain_for_top_plots(
                item.get("domain") or urlparse(cited_url).netloc
            )
            if not domain:
                continue

            if cited_url in retrieved_urls:
                retrieved_cited_counts[domain] = retrieved_cited_counts.get(domain, 0) + 1
            else:
                parametric_cited_counts[domain] = parametric_cited_counts.get(domain, 0) + 1

    domains = sorted(set(retrieved_cited_counts) | set(parametric_cited_counts))
    ranked = [
        (
            domain,
            int(retrieved_cited_counts.get(domain, 0)),
            int(parametric_cited_counts.get(domain, 0)),
            int(retrieved_cited_counts.get(domain, 0))
            + int(parametric_cited_counts.get(domain, 0)),
        )
        for domain in domains
    ]
    ranked.sort(key=lambda item: item[3], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


def plot_replay_web_call_agreement_counts(
    input_path=OUTPUT_DIR / "all_models__prompt_and_web_queries.json",
    base_model_name="gpt-5.3-chat-latest",
    model_names=DEFAULT_MODELS,
    output_dir=PLOT_OUTPUT_DIR,
    output_stem=None,
):
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    from src.utils.figure_style import with_paper_style, styler

    with open(input_path) as f:
        data = json.load(f)

    sample_rows = list(data.values())
    model_names = [model for model in model_names if model != base_model_name]
    other_model_count = len(model_names)
    segment_order = list(range(other_model_count, -1, -1))

    base_called_distribution = {match_count: 0 for match_count in segment_order}
    base_not_called_distribution = {match_count: 0 for match_count in segment_order}

    for row in sample_rows:
        base_called = bool(row.get(base_model_name, []))
        other_calls = [
            bool(row.get(model_name, []))
            for model_name in model_names
        ]
        other_called_count = sum(other_calls)
        other_not_called_count = other_model_count - other_called_count

        if base_called:
            base_called_distribution[other_called_count] += 1
        else:
            base_not_called_distribution[other_not_called_count] += 1

    color_scale = qualitative.Plotly
    if len(color_scale) < len(segment_order):
        multiplier = (len(segment_order) // len(color_scale)) + 1
        color_scale = (color_scale * multiplier)[: len(segment_order)]

    base_label = base_model_name.replace("-latest", "").replace("-2025-08-07", "")
    y_labels = [
        f"{base_label}<br>Did Not Call<br>Web Search",
        f"{base_label}<br>Called<br>Web Search",
    ]

    fig = go.Figure()
    for idx, match_count in enumerate(segment_order):
        not_called_value = base_not_called_distribution[match_count]
        called_value = base_called_distribution[match_count]
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=[not_called_value, called_value],
                orientation="h",
                name=f"{match_count} of {other_model_count} other models",
                marker_color=color_scale[idx],
                text=[
                    str(not_called_value) if not_called_value else "",
                    str(called_value) if called_value else "",
                ],
                textposition="inside",
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Samples",
        yaxis_title="",
        legend_title="Matching Models",
        margin=dict(l=5, r=20, t=100, b=80),
    )
    fig = with_paper_style(fig, config=styler(20, 18))
    fig.update_layout(
        margin=dict(l=5, r=20, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = output_stem or f"replay_{base_model_name}_auto_web_call_venn_agreement_counts"
    fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

    summary = {
        "base_model_name": base_model_name,
        "other_models": model_names,
        "base_called_distribution": base_called_distribution,
        "base_not_called_distribution": base_not_called_distribution,
        "num_samples": len(sample_rows),
    }
    with open(output_dir / f"{file_name}.json", "w") as f:
        json.dump(summary, f, indent=2)


def plot_openai_replay_model_agreement_counts():
    rows = _extract_rows_for_models(OPENAI_REPLAY_MODELS)
    output_path = OUTPUT_DIR / "openai_replay_models__prompt_and_web_queries.json"
    save_prompt_and_model_web_queries(rows, output_path)
    plot_replay_web_call_agreement_counts(
        input_path=output_path,
        base_model_name="gpt-5.3-chat-latest",
        model_names=OPENAI_REPLAY_MODELS,
        output_stem="replay_gpt-5.3-chat-latest_openai_models_auto_web_call_venn_agreement_counts",
    )


def plot_openai_replay_model_call_outcomes(
    input_path=OUTPUT_DIR / "openai_replay_models__prompt_and_web_queries.json",
    base_model_name="gpt-5.3-chat-latest",
    model_names=OPENAI_REPLAY_MODELS,
    output_dir=PLOT_OUTPUT_DIR,
    output_stem="replay_models_auto_web_call_base_gpt-5.3-chat-latest",
):
    import plotly.graph_objects as go

    from src.utils.figure_style import with_paper_style, styler

    sample_calls = {}
    for model_name in model_names:
        replay_path = INPUT_DIR / f"{model_name}.json"
        replay_data = _load_replay_json(replay_path)
        for result_key, row in replay_data.items():
            if not isinstance(row, dict) or row.get("skipped_replay"):
                continue
            payload = row.get("auto")
            if not isinstance(payload, dict):
                continue
            response = payload.get("response") or {}
            provider = _infer_provider(model_name, row)
            called = _has_web_tool_call(provider, response)
            if result_key not in sample_calls:
                sample_calls[result_key] = {}
            sample_calls[result_key][model_name] = called

    common_result_keys = [
        result_key
        for result_key, calls in sample_calls.items()
        if all(model_name in calls for model_name in model_names)
    ]

    comparison_models = [model for model in model_names if model != base_model_name]
    outcome_rows = []

    for model_name in comparison_models:
        base_called_and_model_called = 0
        base_called_and_model_not_called = 0
        base_not_called_and_model_not_called = 0
        base_not_called_and_model_called = 0

        for result_key in common_result_keys:
            base_called = bool(sample_calls[result_key].get(base_model_name, False))
            model_called = bool(sample_calls[result_key].get(model_name, False))
            if base_called and model_called:
                base_called_and_model_called += 1
            elif base_called and not model_called:
                base_called_and_model_not_called += 1
            elif not base_called and not model_called:
                base_not_called_and_model_not_called += 1
            else:
                base_not_called_and_model_called += 1

        outcome_rows.append(
            {
                "model": _model_label(model_name),
                "base_called_and_model_called": base_called_and_model_called,
                "base_called_and_model_not_called": base_called_and_model_not_called,
                "base_not_called_and_model_not_called": base_not_called_and_model_not_called,
                "base_not_called_and_model_called": base_not_called_and_model_called,
            }
        )

    y_labels = [_model_label(model_name) for model_name in comparison_models]
    base_model_label = _model_label(base_model_name)
    outcome_specs = [
        (f"{base_model_label} called, model called", "#2ca02c", "base_called_and_model_called"),
        (f"{base_model_label} called, model did not", "#d62728", "base_called_and_model_not_called"),
        (f"{base_model_label} did not, model did not", "#1f77b4", "base_not_called_and_model_not_called"),
        (f"{base_model_label} did not, model called", "#ff7f0e", "base_not_called_and_model_called"),
    ]

    outcome_lookup = {row["model"]: row for row in outcome_rows}

    fig = go.Figure()
    for outcome_label, color, key in outcome_specs:
        values = [int(outcome_lookup[label][key]) for label in y_labels]
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=values,
                orientation="h",
                name=outcome_label,
                marker_color=color,
                text=[str(value) if value else "" for value in values],
                textposition="inside",
                hovertemplate="%{y}<br>%{fullData.name}: %{x}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Samples",
        yaxis_title="Replay Model",
        legend_title="Outcome",
        margin=dict(l=5, r=20, t=100, b=80),
    )
    fig = with_paper_style(fig, config=styler(20, 18))
    fig.update_layout(
        margin=dict(l=5, r=20, t=100, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(f"{output_dir}/{output_stem}.pdf", format="pdf")
    with open(output_dir / f"{output_stem}_summary.json", "w") as f:
        json.dump(outcome_rows, f, indent=2)


def plot_cross_platform_replay_model_call_outcomes(
    input_path=OUTPUT_DIR / "all_models__prompt_and_web_queries.json",
    base_model_name="gpt-5.3-chat-latest",
    model_names=DEFAULT_MODELS,
    output_dir=PLOT_OUTPUT_DIR,
    output_stem="replay_models_cross_platform_auto_web_call_base_gpt-5.3-chat-latest",
):
    plot_openai_replay_model_call_outcomes(
        input_path=input_path,
        base_model_name=base_model_name,
        model_names=model_names,
        output_dir=output_dir,
        output_stem=output_stem,
    )


def plot_openai_replay_model_outcome_trigger_heatmaps(
    metadata_path=REPLAY_SAMPLE_CHARACTERIZATION_PATH,
    base_model_name="gpt-5.3-chat-latest",
    model_names=OPENAI_REPLAY_MODELS,
    output_dir=PLOT_OUTPUT_DIR,
    output_stem="replay_models_auto_web_call_outcomes_by_primary_trigger_base_gpt-5.3-chat-latest",
):
    import plotly.graph_objects as go

    from src.utils.figure_style import with_paper_style, styler

    rows = _extract_rows_for_models(model_names)
    grouped_samples = _group_rows_by_sample(rows)
    trigger_by_turn, ordered_triggers = _load_primary_triggers(metadata_path)
    trigger_labels = ordered_triggers + [NO_CHARACTERIZATION_LABEL]

    base_model_label = _model_label(base_model_name)
    comparison_models = [model for model in model_names if model != base_model_name]
    outcome_specs = [
        (
            "base_not_called_and_model_called",
            f"{base_model_label} did not<br>model called",
        ),
        (
            "base_not_called_and_model_not_called",
            f"{base_model_label} did not<br>model did not",
        ),
        (
            "base_called_and_model_not_called",
            f"{base_model_label} called<br>model did not",
        ),
        (
            "base_called_and_model_called",
            f"{base_model_label} called<br>model called",
        ),
    ]

    summary = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in comparison_models:
        model_label = _model_label(model_name)
        outcome_specs = [
            (
                "base_not_called_and_model_called",
                f"{base_model_label} did not<br>{model_label} called",
            ),
            (
                "base_not_called_and_model_not_called",
                f"{base_model_label} did not<br>{model_label} did not",
            ),
            (
                "base_called_and_model_not_called",
                f"{base_model_label} called<br>{model_label} did not",
            ),
            (
                "base_called_and_model_called",
                f"{base_model_label} called<br>{model_label} called",
            ),
        ]
        counts = {
            outcome_key: {trigger: 0 for trigger in trigger_labels}
            for outcome_key, _ in outcome_specs
        }

        for sample in grouped_samples.values():
            base_called = bool(sample["models"].get(base_model_name, False))
            model_called = bool(sample["models"].get(model_name, False))
            outcome_key = _outcome_key(base_called, model_called)
            trigger = trigger_by_turn.get(
                (sample["conv_id"], sample["turn_id"]),
                NO_CHARACTERIZATION_LABEL,
            )
            counts[outcome_key][trigger] = counts[outcome_key].get(trigger, 0) + 1

        z_values = []
        text_values = []
        summary[model_name] = {}

        for outcome_key, outcome_label in outcome_specs:
            row_counts = [counts[outcome_key].get(trigger, 0) for trigger in trigger_labels]
            total = sum(row_counts)
            row_percentages = [
                (count / total * 100.0) if total else 0.0 for count in row_counts
            ]
            z_values.append(row_percentages)
            text_values.append(
                [
                    f"{count}<br>{percentage:.0f}%"
                    if count
                    else ""
                    for count, percentage in zip(row_counts, row_percentages)
                ]
            )
            summary[model_name][outcome_key] = {
                "outcome_label": outcome_label,
                "total_samples": total,
                "trigger_counts": {
                    trigger: {
                        "count": count,
                        "percentage": percentage,
                    }
                    for trigger, count, percentage in zip(
                        trigger_labels,
                        row_counts,
                        row_percentages,
                    )
                },
            }

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=z_values,
                    x=trigger_labels,
                    y=[label for _, label in outcome_specs],
                    colorscale="Blues",
                    zmin=0,
                    zmax=100,
                    xgap=2,
                    ygap=2,
                    text=text_values,
                    texttemplate="%{text}",
                    hovertemplate=(
                        "Outcome: %{y}<br>"
                        "Trigger: %{x}<br>"
                        "Row share: %{z:.1f}%<extra></extra>"
                    ),
                    colorbar=dict(title="Row %"),
                )
            ]
        )

        fig.update_layout(
            title=f"{model_label}: call outcomes by primary trigger",
            xaxis_title="Primary Trigger",
            yaxis_title="Outcome",
            plot_bgcolor="black",
            margin=dict(t=10),
        )
        fig.update_xaxes(tickangle=35)
        fig = with_paper_style(fig, config=styler(18, 16))
        fig.update_layout(margin=dict(t=10))

        model_slug = str(model_name).replace(".", "-")
        fig.write_image(
            output_dir
            / f"{output_stem}__{model_slug}.pdf",
            format="pdf",
        )

    with open(output_dir / f"{output_stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def plot_replay_pair_outcome_trigger_heatmap(
    metadata_path=REPLAY_SAMPLE_CHARACTERIZATION_PATH,
    base_model_name="gpt-4.1-mini-2025-04-14",
    comparison_model_name="o4-mini-2025-04-16",
    model_names=OPENAI_REPLAY_MODELS,
    output_dir=PLOT_OUTPUT_DIR,
    output_stem="replay_pair_auto_web_call_outcomes_by_primary_trigger",
    trigger_model_name=None,
):
    import plotly.graph_objects as go

    from src.utils.figure_style import with_paper_style, styler

    rows = _extract_rows_for_models(model_names)
    grouped_samples = _group_rows_by_sample(rows)
    if trigger_model_name is None:
        trigger_model_name = comparison_model_name
    trigger_by_turn, ordered_triggers = _load_primary_triggers(
        metadata_path,
        judge_model=trigger_model_name,
    )
    trigger_labels = ordered_triggers + [NO_CHARACTERIZATION_LABEL]

    base_model_label = _model_label(base_model_name)
    comparison_model_label = _model_label(comparison_model_name)
    outcome_specs = [
        (
            "base_not_called_and_model_called",
            f"{base_model_label} did not<br>{comparison_model_label} called",
        ),
        (
            "base_not_called_and_model_not_called",
            f"{base_model_label} did not<br>{comparison_model_label} did not",
        ),
        (
            "base_called_and_model_not_called",
            f"{base_model_label} called<br>{comparison_model_label} did not",
        ),
        (
            "base_called_and_model_called",
            f"{base_model_label} called<br>{comparison_model_label} called",
        ),
    ]

    counts = {
        outcome_key: {trigger: 0 for trigger in trigger_labels}
        for outcome_key, _ in outcome_specs
    }

    for sample in grouped_samples.values():
        base_called = bool(sample["models"].get(base_model_name, False))
        model_called = bool(sample["models"].get(comparison_model_name, False))
        outcome_key = _outcome_key(base_called, model_called)
        trigger = trigger_by_turn.get(
            (sample["conv_id"], sample["turn_id"]),
            NO_CHARACTERIZATION_LABEL,
        )
        counts[outcome_key][trigger] = counts[outcome_key].get(trigger, 0) + 1

    z_values = []
    text_values = []
    summary = {
        "base_model": base_model_name,
        "comparison_model": comparison_model_name,
        "trigger_model": trigger_model_name,
        "outcomes": {},
    }

    for outcome_key, outcome_label in outcome_specs:
        row_counts = [counts[outcome_key].get(trigger, 0) for trigger in trigger_labels]
        total = sum(row_counts)
        row_percentages = [
            (count / total * 100.0) if total else 0.0 for count in row_counts
        ]
        z_values.append(row_percentages)
        text_values.append(
            [
                f"{count}<br>{percentage:.0f}%"
                if count
                else ""
                for count, percentage in zip(row_counts, row_percentages)
            ]
        )
        summary["outcomes"][outcome_key] = {
            "outcome_label": outcome_label,
            "total_samples": total,
            "trigger_counts": {
                trigger: {
                    "count": count,
                    "percentage": percentage,
                }
                for trigger, count, percentage in zip(
                    trigger_labels,
                    row_counts,
                    row_percentages,
                )
            },
        }

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=trigger_labels,
                y=[label for _, label in outcome_specs],
                colorscale="Blues",
                zmin=0,
                zmax=100,
                xgap=2,
                ygap=2,
                text=text_values,
                texttemplate="%{text}",
                hovertemplate=(
                    "Outcome: %{y}<br>"
                    "Trigger: %{x}<br>"
                    "Row share: %{z:.1f}%<extra></extra>"
                ),
                colorbar=dict(title="Row %"),
            )
        ]
    )

    fig.update_layout(
        title=f"{base_model_label} vs {comparison_model_label}: call outcomes by primary trigger",
        xaxis_title="Primary Trigger",
        yaxis_title="Outcome",
        plot_bgcolor="black",
        margin=dict(t=10),
    )
    fig.update_xaxes(tickangle=35)
    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_layout(margin=dict(t=10))

    output_dir.mkdir(parents=True, exist_ok=True)
    base_slug = str(base_model_name).replace(".", "-")
    comparison_slug = str(comparison_model_name).replace(".", "-")
    fig.write_image(
        output_dir / f"{output_stem}__{base_slug}__vs__{comparison_slug}.pdf",
        format="pdf",
    )
    with open(
        output_dir / f"{output_stem}__{base_slug}__vs__{comparison_slug}_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def plot_openai_replay_dev_prompt_web_call_heatmap(
    replay_models=None,
    prompt_models=None,
    input_dir=INPUT_DIR,
    output_dir=PLOT_OUTPUT_DIR,
    output_stem="replay_openai_dev_prompt_web_call_heatmap",
):
    import plotly.graph_objects as go

    from src.utils.figure_style import with_paper_style, styler

    if replay_models is None:
        replay_models = [
            "gpt-5.3-chat-latest",
            "gpt-4.1-mini-2025-04-14",
            "o4-mini-2025-04-16",
        ]
    if prompt_models is None:
        prompt_models = [
            "gpt-5.3-chat-latest",
            "gpt-4.1-mini-2025-04-14",
            "o4-mini-2025-04-16",
        ]

    def _dev_prompt_suffix(prompt_model_name):
        if prompt_model_name == "gpt-4.1-mini-2025-04-14":
            return "gpt-4.1-mini"
        if prompt_model_name == "o4-mini-2025-04-16":
            return "o4-mini"
        return prompt_model_name

    def _replay_file_for_combo(replay_model_name, prompt_model_name):
        if replay_model_name == prompt_model_name:
            return input_dir / f"{replay_model_name}.json"
        suffix = _dev_prompt_suffix(prompt_model_name)
        candidates = sorted(
            input_dir.glob(f"{replay_model_name}_*dev_prompt_{suffix}*.json")
        )
        if not candidates:
            return input_dir / f"{replay_model_name}_dev_prompt_{suffix}.json"
        return candidates[0]

    def _count_web_call_samples(replay_path, replay_model_name):
        replay_data = _load_replay_json(replay_path)
        sample_calls = {}
        for _, row in replay_data.items():
            if not isinstance(row, dict) or row.get("skipped_replay"):
                continue
            result_key = row.get("result_key")
            if not result_key:
                continue
            payload = row.get("auto")
            if payload is None:
                continue
            response = payload.get("response") or {}
            provider = _infer_provider(replay_model_name, row)
            sample_calls[result_key] = _has_web_tool_call(provider, response)
        return sample_calls

    combo_calls = {}
    common_result_keys = None
    for replay_model_name in replay_models:
        for prompt_model_name in prompt_models:
            replay_path = _replay_file_for_combo(replay_model_name, prompt_model_name)
            if not replay_path.exists():
                raise FileNotFoundError(f"Missing replay file for combo: {replay_path}")
            sample_calls = _count_web_call_samples(replay_path, replay_model_name)
            combo_calls[(replay_model_name, prompt_model_name)] = {
                "replay_path": replay_path,
                "sample_calls": sample_calls,
            }
            result_keys = set(sample_calls.keys())
            common_result_keys = (
                result_keys
                if common_result_keys is None
                else common_result_keys & result_keys
            )

    if common_result_keys is None:
        common_result_keys = set()

    z_values = []
    text_values = []
    summary = {}
    for replay_model_name in replay_models:
        row_counts = []
        row_text = []
        summary[replay_model_name] = {}
        for prompt_model_name in prompt_models:
            combo_info = combo_calls[(replay_model_name, prompt_model_name)]
            replay_path = combo_info["replay_path"]
            sample_calls = combo_info["sample_calls"]
            count = sum(
                1
                for result_key in common_result_keys
                if sample_calls.get(result_key, False)
            )
            row_counts.append(count)
            row_text.append(str(count))
            summary[replay_model_name][prompt_model_name] = {
                "web_call_samples": int(count),
                "replay_path": str(replay_path),
                "num_common_samples": int(len(common_result_keys)),
            }
        z_values.append(row_counts)
        text_values.append(row_text)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=[_model_label(model_name) for model_name in prompt_models],
                y=[_model_label(model_name) for model_name in replay_models],
                text=text_values,
                texttemplate="%{text}",
                textfont=dict(size=30),
                colorscale="Blues",
                hovertemplate=(
                    "Replay model: %{y}<br>"
                    "Developer prompt: %{x}<br>"
                    "Web-calling samples: %{z}<extra></extra>"
                ),
                xgap=2,
                ygap=2,
                colorbar=dict(title="Count"),
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Developer Prompt Model",
        yaxis_title="Replay Model",
        margin=dict(l=80, r=40, t=30, b=80),
        plot_bgcolor="black",
    )
    fig = with_paper_style(fig, config=styler(22, 18), legend_pos=None)
    fig.update_layout(
        margin=dict(l=80, r=40, t=30, b=80),
        plot_bgcolor="black",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_dir / f"{output_stem}.pdf", format="pdf")
    with open(output_dir / f"{output_stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def plot_replay_query_term_count_trends_over_time(
    remove_stopwords=False,
    model_names=DEFAULT_MODELS,
    output_dir=PLOT_OUTPUT_DIR / "query_complexity",
):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from src.web_search_decision.chatgpt_extraction import load_web_data_from_file, load_whole_data_from_file
    from src.utils.figure_style import with_paper_style, styler

    iteration_bucket_order = ["1", "2", "3+"]

    def _timeline_iteration_bucket(num_iterations):
        if num_iterations <= 1:
            return "1"
        if num_iterations == 2:
            return "2"
        return "3+"

    def _sorted_cdf(values):
        sorted_values = np.sort(np.asarray(values, dtype=float))
        cdf_values = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
        return sorted_values, cdf_values

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

    def _build_time_lookup():
        whole_df = load_whole_data_from_file(fmt="pkl").copy()
        web_df = load_web_data_from_file(fmt="pkl").copy()
        df = pd.concat([whole_df, web_df], ignore_index=True)
        df["conv_id"] = df["conv_id"].astype(str)
        df["turn_id"] = df["turn_id"].astype(str)
        df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
        df = df.dropna(subset=["conv_id", "turn_id"])
        df = df.sort_values("time").drop_duplicates(
            subset=["conv_id", "turn_id"], keep="last"
        )
        return {
            (row["conv_id"], row["turn_id"]): row["time"]
            for _, row in df.iterrows()
            if pd.notna(row["time"])
        }

    time_lookup = _build_time_lookup()
    rows = _extract_rows_for_models(model_names)
    rows = _filter_rows_to_common_samples(rows, model_names)
    rows = _filter_rows_to_samples_with_web_calls_for_all_models(rows, model_names)

    metrics_by_model = {}
    palette = px.colors.qualitative.Plotly
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_rows = [row for row in rows if row.get("model") == model_name]
        metrics = {
            "terms_per_query": [],
            "terms_per_prompt": [],
            "terms_per_prompt_web_call": [],
            "avg_query_terms_per_prompt": [],
            "total_query_terms_per_prompt": [],
            "total_queries_per_prompt": [],
            "retrieved_urls_per_prompt": [],
            "retrieved_urls_per_web_query": [],
            "iterations_per_prompt": [],
            "iteration_timeline_rows": [],
        }

        for row in model_rows:
            if bool(row.get("has_web_tool_call", False)):
                metrics["terms_per_prompt_web_call"].append(
                    _count_terms_local(row.get("user_prompt", ""), remove_stopwords)
                )

            web_query_groups = _clean_web_query_groups(row.get("web_queries", []))
            if not web_query_groups:
                continue

            flat_web_queries = [query for group in web_query_groups for query in group]
            if not flat_web_queries:
                continue

            web_query_term_counts = [
                _count_terms_local(query, remove_stopwords) for query in flat_web_queries
            ]
            if not web_query_term_counts:
                continue

            metrics["terms_per_query"] += web_query_term_counts
            metrics["terms_per_prompt"].append(
                _count_terms_local(row.get("user_prompt", ""), remove_stopwords)
            )
            metrics["avg_query_terms_per_prompt"].append(
                float(np.mean(web_query_term_counts))
            )
            metrics["total_query_terms_per_prompt"].append(
                float(np.sum(web_query_term_counts))
            )
            metrics["total_queries_per_prompt"].append(len(flat_web_queries))
            retrieved_url_count = _count_nested_sources(
                row.get("sources_retrieved", []) or []
            )
            metrics["retrieved_urls_per_prompt"].append(float(retrieved_url_count))
            if retrieved_url_count > 0:
                per_query_retrieved_count = retrieved_url_count / len(flat_web_queries)
                metrics["retrieved_urls_per_web_query"] += [
                    float(per_query_retrieved_count)
                ] * len(flat_web_queries)
            num_iterations = int(len(web_query_groups))
            metrics["iterations_per_prompt"].append(num_iterations)

            row_time = time_lookup.get(
                (str(row.get("conv_id", "") or ""), str(row.get("turn_id", "") or ""))
            )
            if pd.notna(row_time):
                metrics["iteration_timeline_rows"].append(
                    {
                        "month": row_time.to_period("M").to_timestamp(),
                        "iteration_bucket": _timeline_iteration_bucket(num_iterations),
                    }
                )

        if metrics["terms_per_query"] and metrics["terms_per_prompt_web_call"]:
            metrics_by_model[model_name] = metrics

    if not metrics_by_model:
        print("No valid replay rows found for query-complexity plots.")
        return {}

    def _plot_cdf_by_model(
        metric_key,
        *,
        value_col,
        xaxis_title,
        file_name,
        hover_label,
        x_fmt=".2f",
        xaxis_config=None,
    ):
        fig = go.Figure()
        points_by_model = {}

        for idx, model_name in enumerate(model_names):
            if model_name not in metrics_by_model:
                continue

            values = metrics_by_model[model_name][metric_key]
            if not values:
                continue

            sorted_values, cdf_values = _sorted_cdf(values)
            points_by_model[model_name] = pd.DataFrame(
                {value_col: sorted_values, "cdf": cdf_values}
            ).to_dict(orient="records")

            color = palette[idx % len(palette)]
            display_name = _model_label(model_name)
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
            "margin": dict(t=95),
            "legend": dict(
                orientation="h",
                yanchor="bottom",
                y=1.16,
                xanchor="left",
                x=0,
                entrywidthmode="fraction",
                entrywidth=0.48,
            ),
        }
        if xaxis_config is not None:
            xaxis_settings = dict(xaxis_config)
            range_values = xaxis_settings.get("range")
            if isinstance(range_values, (list, tuple)) and len(range_values) == 2:
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
                        if tick_text:
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
        # fig.write_html(output_dir / f"{file_name}.html")
        fig = with_paper_style(fig, config=styler(26, 23))
        fig.update_layout(
            margin=dict(t=95),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.16,
                xanchor="left",
                x=0,
                entrywidthmode="fraction",
                entrywidth=0.5,
            ),
        )
        fig.write_image(output_dir / f"{file_name}.pdf", format="pdf")
        return points_by_model

    web_query_cdf_points_by_model = _plot_cdf_by_model(
        "terms_per_query",
        value_col="web_query_terms",
        xaxis_title="Number of Terms",
        file_name="replay_web_query_terms_cdf",
        hover_label="Web query terms",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    user_prompt_cdf_points_by_model = _plot_cdf_by_model(
        "terms_per_prompt_web_call",
        value_col="user_prompt_terms",
        xaxis_title="Number of Terms",
        file_name="replay_user_prompt_terms_cdf",
        hover_label="User prompt terms",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    avg_web_query_terms_per_prompt_cdf_points_by_model = _plot_cdf_by_model(
        "avg_query_terms_per_prompt",
        value_col="avg_web_query_terms_per_prompt",
        xaxis_title="Average Number of Web Query Terms Per User Prompt",
        file_name="replay_avg_web_query_terms_per_prompt_cdf",
        hover_label="Avg web query terms per prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    total_web_query_terms_per_prompt_cdf_points_by_model = _plot_cdf_by_model(
        "total_query_terms_per_prompt",
        value_col="total_web_query_terms_per_prompt",
        xaxis_title="Total Number of Web Query Terms Per User Prompt",
        file_name="replay_total_web_query_terms_per_prompt_cdf",
        hover_label="Total web query terms per prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    total_web_queries_per_prompt_cdf_points_by_model = _plot_cdf_by_model(
        "total_queries_per_prompt",
        value_col="total_web_queries_per_prompt",
        xaxis_title="Number of Web Queries Per User Prompt",
        file_name="replay_total_web_queries_per_prompt_cdf",
        hover_label="Total web queries per prompt",
        x_fmt=".0f",
        xaxis_config=dict(range=[0, 10]),
    )
    retrieved_urls_per_prompt_cdf_points_by_model = _plot_cdf_by_model(
        "retrieved_urls_per_prompt",
        value_col="retrieved_urls_per_prompt",
        xaxis_title="#Search Result URLs Per User Prompt",
        file_name="replay_retrieved_urls_per_prompt_cdf",
        hover_label="Retrieved URLs per user prompt",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    retrieved_urls_per_web_query_cdf_points_by_model = _plot_cdf_by_model(
        "retrieved_urls_per_web_query",
        value_col="retrieved_urls_per_web_query",
        xaxis_title="#Search Result URLs Per Web Query",
        file_name="replay_retrieved_urls_per_web_query_cdf",
        hover_label="Retrieved URLs per web query",
        x_fmt=".2f",
        xaxis_config=dict(range=[0, 20]),
    )
    iterations_per_prompt_cdf_points_by_model = _plot_cdf_by_model(
        "iterations_per_prompt",
        value_col="iterations_per_prompt",
        xaxis_title="Number of Iterations Per User Prompt",
        file_name="replay_iterations_per_prompt_cdf",
        hover_label="Iterations per prompt",
        x_fmt=".0f",
        xaxis_config=dict(range=[0, 10]),
    )

    timeline_points_by_model = {}
    timeline_plot_files_by_model = {}
    iteration_bucket_display = {
        "1": "1 iteration",
        "2": "2 iterations",
        "3+": "3+ iterations",
    }
    for model_name in model_names:
        if model_name not in metrics_by_model:
            continue

        iteration_timeline_df = pd.DataFrame(
            metrics_by_model[model_name]["iteration_timeline_rows"]
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

        display_name = _model_label(model_name)
        timeline_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Share of User Prompts",
            title=f"Replay User Prompts by Iteration Bucket Over Time ({display_name})",
            xaxis=dict(
                tickmode="linear",
                dtick="M1",
                tickformat="%b %Y",
                tickangle=-45,
            ),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            margin=dict(b=90),
        )
        timeline_file_name = f"replay_web_prompts_by_iteration_over_time_{model_name}"
        timeline_fig.write_html(output_dir / f"{timeline_file_name}.html")
        timeline_fig = with_paper_style(timeline_fig, config=styler(18, 16))
        timeline_fig.write_image(output_dir / f"{timeline_file_name}.pdf", format="pdf")

        timeline_records = monthly_iteration_counts.copy()
        timeline_records["month"] = timeline_records["month"].dt.strftime("%Y-%m")
        timeline_records["iteration_bucket"] = timeline_records["iteration_bucket"].astype(
            str
        )
        timeline_points_by_model[model_name] = timeline_records.to_dict(orient="records")
        timeline_plot_files_by_model[model_name] = timeline_file_name

    summary_by_model = {}
    for model_name, metrics in metrics_by_model.items():
        summary_by_model[model_name] = {
            "num_web_queries": int(len(metrics["terms_per_query"])),
            "num_prompts": int(len(metrics["terms_per_prompt"])),
            "mean_web_query_terms": float(np.mean(metrics["terms_per_query"])),
            "median_web_query_terms": float(np.median(metrics["terms_per_query"])),
            "mean_user_prompt_terms": float(np.mean(metrics["terms_per_prompt"])),
            "median_user_prompt_terms": float(np.median(metrics["terms_per_prompt"])),
            "mean_avg_web_query_terms_per_prompt": float(
                np.mean(metrics["avg_query_terms_per_prompt"])
            ),
            "median_avg_web_query_terms_per_prompt": float(
                np.median(metrics["avg_query_terms_per_prompt"])
            ),
            "mean_total_web_query_terms_per_prompt": float(
                np.mean(metrics["total_query_terms_per_prompt"])
            ),
            "median_total_web_query_terms_per_prompt": float(
                np.median(metrics["total_query_terms_per_prompt"])
            ),
            "mean_total_web_queries_per_prompt": float(
                np.mean(metrics["total_queries_per_prompt"])
            ),
            "median_total_web_queries_per_prompt": float(
                np.median(metrics["total_queries_per_prompt"])
            ),
            "mean_retrieved_urls_per_prompt": float(
                np.mean(metrics["retrieved_urls_per_prompt"])
            )
            if metrics["retrieved_urls_per_prompt"]
            else 0.0,
            "median_retrieved_urls_per_prompt": float(
                np.median(metrics["retrieved_urls_per_prompt"])
            )
            if metrics["retrieved_urls_per_prompt"]
            else 0.0,
            "mean_retrieved_urls_per_web_query": float(
                np.mean(metrics["retrieved_urls_per_web_query"])
            )
            if metrics["retrieved_urls_per_web_query"]
            else 0.0,
            "median_retrieved_urls_per_web_query": float(
                np.median(metrics["retrieved_urls_per_web_query"])
            )
            if metrics["retrieved_urls_per_web_query"]
            else 0.0,
            "mean_iterations_per_prompt": float(np.mean(metrics["iterations_per_prompt"])),
            "median_iterations_per_prompt": float(
                np.median(metrics["iterations_per_prompt"])
            ),
        }

    with open(output_dir / "replay_query_complexity_summary.json", "w") as f:
        json.dump(
            {
                "models_plotted": [m for m in model_names if m in metrics_by_model],
                "summary_by_model": summary_by_model,
                "web_query_cdf_points_by_model": web_query_cdf_points_by_model,
                "user_prompt_cdf_points_by_model": user_prompt_cdf_points_by_model,
                "avg_web_query_terms_per_prompt_cdf_points_by_model": avg_web_query_terms_per_prompt_cdf_points_by_model,
                "total_web_query_terms_per_prompt_cdf_points_by_model": total_web_query_terms_per_prompt_cdf_points_by_model,
                "total_web_queries_per_prompt_cdf_points_by_model": total_web_queries_per_prompt_cdf_points_by_model,
                "retrieved_urls_per_prompt_cdf_points_by_model": retrieved_urls_per_prompt_cdf_points_by_model,
                "retrieved_urls_per_web_query_cdf_points_by_model": retrieved_urls_per_web_query_cdf_points_by_model,
                "iterations_per_prompt_cdf_points_by_model": iterations_per_prompt_cdf_points_by_model,
                "web_prompts_by_iteration_over_time_points_by_model": timeline_points_by_model,
                "web_prompts_by_iteration_over_time_plot_files_by_model": timeline_plot_files_by_model,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "models_plotted": [m for m in model_names if m in metrics_by_model],
        "summary_by_model": summary_by_model,
        "web_query_cdf_points_by_model": web_query_cdf_points_by_model,
        "user_prompt_cdf_points_by_model": user_prompt_cdf_points_by_model,
        "avg_web_query_terms_per_prompt_cdf_points_by_model": avg_web_query_terms_per_prompt_cdf_points_by_model,
        "total_web_query_terms_per_prompt_cdf_points_by_model": total_web_query_terms_per_prompt_cdf_points_by_model,
        "total_web_queries_per_prompt_cdf_points_by_model": total_web_queries_per_prompt_cdf_points_by_model,
        "retrieved_urls_per_prompt_cdf_points_by_model": retrieved_urls_per_prompt_cdf_points_by_model,
        "retrieved_urls_per_web_query_cdf_points_by_model": retrieved_urls_per_web_query_cdf_points_by_model,
        "iterations_per_prompt_cdf_points_by_model": iterations_per_prompt_cdf_points_by_model,
        "web_prompts_by_iteration_over_time_points_by_model": timeline_points_by_model,
        "web_prompts_by_iteration_over_time_plot_files_by_model": timeline_plot_files_by_model,
    }


def plot_replay_parallel_queries_by_query_reformulations(
    model_names=DEFAULT_MODELS,
    output_dir=PLOT_OUTPUT_DIR / "query_reformulations",
):
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from src.utils.figure_style import with_paper_style, styler

    platform_color_map = {
        "gpt-5.3-chat-latest": "#636EFA",
        "claude-sonnet-4-6": "#EF553B",
        "grok-4.3": "#00CC96",
        "deepseek-v4-flash": "#AB63FA",
    }
    platform_pattern_map = {
        "gpt-5.3-chat-latest": ".",
        "claude-sonnet-4-6": "+",
        "grok-4.3": "x",
        "deepseek-v4-flash": "/",
        "o4-mini-2025-04-16": ".",
        "gpt-4.1-mini-2025-04-14": "+",
    }

    bucket_order = ["1", "2", "3+"]
    bucket_colors = {
        "1": "#636EFA",
        "2": "#EF553B",
        "3+": "#00CC96",
    }
    model_slug = _model_subset_slug(model_names)

    def _bucket_count(value):
        return "3+" if value >= 3 else str(value)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _extract_rows_for_models(model_names)

    summaries = {}
    subplot_titles = []
    subplot_payloads = []
    for model_name in model_names:
        model_rows = [row for row in rows if row.get("model") == model_name]
        count_n_hops = {}
        parallel_query_counts = {
            n_hops_bucket: {parallel_bucket: 0 for parallel_bucket in bucket_order}
            for n_hops_bucket in bucket_order
        }

        for row in model_rows:
            web_query_groups = _clean_web_query_groups(row.get("web_queries", []))
            n_hops = len(web_query_groups)
            if n_hops == 0:
                continue

            count_n_hops[n_hops] = count_n_hops.get(n_hops, 0) + 1
            max_parallel_queries = max((len(group) for group in web_query_groups), default=0)
            if max_parallel_queries >= 1:
                parallel_query_counts[_bucket_count(n_hops)][
                    _bucket_count(max_parallel_queries)
                ] += 1

        count_n_hops_sum = sum(count_n_hops.values())
        if count_n_hops_sum == 0:
            print(f"No web query loops to plot for `{model_name}`.")
            continue

        fig = go.Figure()
        for parallel_bucket in bucket_order:
            y = [
                round(
                    100
                    * parallel_query_counts[n_hops_bucket][parallel_bucket]
                    / count_n_hops_sum,
                    2,
                )
                for n_hops_bucket in bucket_order
            ]
            fig.add_trace(
                go.Bar(
                    x=bucket_order,
                    y=y,
                    name=parallel_bucket,
                    marker_color=bucket_colors[parallel_bucket],
                    text=[f"{value:.1f}%" if value > 0 else "" for value in y],
                    textposition="outside",
                    textfont=dict(size=18),
                    hovertemplate=(
                        "Query formulations: %{x}<br>"
                        f"Max Fan-out queries: {parallel_bucket}<br>"
                        "Turns: %{customdata}<br>"
                        "Share: %{y:.2f}%<extra></extra>"
                    ),
                    customdata=[
                        parallel_query_counts[n_hops_bucket][parallel_bucket]
                        for n_hops_bucket in bucket_order
                    ],
                )
            )

        max_breakdown_y = max(
            [
                100 * count / count_n_hops_sum
                for n_hops_bucket in bucket_order
                for count in parallel_query_counts[n_hops_bucket].values()
            ]
            or [0]
        )
        display_name = _model_label(model_name)
        fig.update_layout(
            barmode="group",
            xaxis_title="Number of Query Formulations",
            yaxis_title="Turns (%)",
            yaxis=dict(range=[0, max_breakdown_y * 1.25 if max_breakdown_y else 1]),
            legend_title_text="Max Fan-out Queries",
            title=f"{display_name}: parallel queries by query reformulations",
            margin=dict(t=90),
        )
        fig.update_yaxes(ticksuffix="%")
        fig = with_paper_style(fig, config=styler(18, 16))
        fig.update_layout(margin=dict(t=90))

        model_slug = str(model_name).replace(".", "-")
        fig.write_image(
            output_dir
            / f"parallel_queries_by_query_reformulations__{model_slug}.pdf",
            format="pdf",
        )

        subplot_titles.append(display_name)
        subplot_payloads.append(
            {
                "model_name": model_name,
                "display_name": display_name,
                "parallel_query_counts": parallel_query_counts,
                "count_n_hops_sum": count_n_hops_sum,
                "max_breakdown_y": max_breakdown_y,
            }
        )

        summaries[model_name] = {
            "num_web_call_prompts": int(count_n_hops_sum),
            "count_n_hops": {
                _bucket_count(n_hops): int(sum_count)
                for n_hops, sum_count in count_n_hops.items()
            },
            "parallel_query_counts": parallel_query_counts,
        }

    if subplot_payloads:
        num_panels = len(subplot_payloads)
        num_cols = 2 if num_panels > 1 else 1
        num_rows = int(np.ceil(num_panels / num_cols))
        combined_fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[item["display_name"] for item in subplot_payloads],
            vertical_spacing=0.2,
            horizontal_spacing=0.08,
        )
        combined_fig.update_annotations(font_size=22)
        legend_shown = set()
        for panel_idx, panel in enumerate(subplot_payloads):
            row_idx = panel_idx // num_cols + 1
            col_idx = panel_idx % num_cols + 1
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
                        textfont=dict(size=32),
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
                range=[0, panel["max_breakdown_y"] * 1.25 if panel["max_breakdown_y"] else 1],
                ticksuffix="%",
                row=row_idx,
                col=col_idx,
            )
            combined_fig.update_xaxes(
                row=row_idx,
                col=col_idx,
            )

        combined_fig.update_layout(
            barmode="group",
            legend_title_text="Max Fan-out Queries",
            margin=dict(r=5, l=100),
            title="Parallel Queries by Query Reformulations",
        )
        combined_fig.add_annotation(
            text="Number of Iterations",
            x=0.5,
            y=-0.05,
            xref="paper",
            yref="paper",
            yshift=-55,
            showarrow=False,
            font=dict(size=24),
        )
        combined_fig.add_annotation(
            text="Turns (%)",
            x=-0.05,
            y=0.5,
            xref="paper",
            yref="paper",
            xshift=-70,
            textangle=-90,
            showarrow=False,
            font=dict(size=24),
        )
        combined_fig = with_paper_style(combined_fig, config=styler(22, 22))
        # combined_fig.update_layout(margin=dict(t=130, b=90, l=90))
        combined_fig.write_image(
            output_dir
            / f"parallel_queries_by_query_reformulations__{model_slug}__all_models.pdf",
            format="pdf",
        )

    with open(
        output_dir
        / f"parallel_queries_by_query_reformulations_summary__{model_slug}.json",
        "w",
    ) as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    return summaries


def plot_replay_top_domains(
    separate_cited_external_internal=True,
    model_names=DEFAULT_MODELS,
    output_dir=PLOT_OUTPUT_DIR / "source_selection",
    common_samples_only=True,
    common_model_names=None,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from src.utils.figure_style import with_paper_style, styler

    platform_color_map = {
        "gpt-5.3-chat-latest": "#636EFA",
        "claude-sonnet-4-6": "#EF553B",
        "grok-4.3": "#00CC96",
        "deepseek-v4-flash": "#AB63FA",
    }
    platform_pattern_map = {
        "gpt-5.3-chat-latest": ".",
        "claude-sonnet-4-6": "+",
        "grok-4.3": "x",
        "deepseek-v4-flash": "/",
    }
    common_filter_model_names = (
        list(common_model_names)
        if common_model_names is not None
        else list(model_names)
    )

    def _build_model_domain_figure(model_name, model_rows):
        subplot_titles = [
            "Top Search Results Domains",
            "Top Cited Search Results Domains",
            "Top Cited Parametric Domains",
        ] if separate_cited_external_internal else [
            "Top Search Results Domains",
            "Top Cited Domains",
        ]

        fig = make_subplots(
            rows=len(subplot_titles),
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.3,
        )

        retrieved_ranked = _domain_counter_from_rows(
            model_rows, "sources_retrieved", top_k=10
        )
        cited_ranked = _domain_counter_from_rows(model_rows, "sources_cited", top_k=10)
        split_ranked = _cited_domain_counter_split_from_rows(model_rows, top_k=None)

        retrieved_total = sum(
            count
            for _, count in _domain_counter_from_rows(
                model_rows, "sources_retrieved", top_k=None
            )
        )
        cited_total = sum(
            count
            for _, count in _domain_counter_from_rows(
                model_rows, "sources_cited", top_k=None
            )
        )
        external_total = sum(
            item[1]
            for item in _cited_domain_counter_split_from_rows(model_rows, top_k=None)
        )
        internal_total = sum(
            item[2]
            for item in _cited_domain_counter_split_from_rows(model_rows, top_k=None)
        )

        if retrieved_total > 0 and retrieved_ranked:
            fig.add_trace(
                go.Bar(
                    x=[domain for domain, _ in retrieved_ranked],
                    y=[count / retrieved_total for _, count in retrieved_ranked],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        if separate_cited_external_internal:
            external_ranked = sorted(
                [item for item in split_ranked if item[1] > 0],
                key=lambda item: item[1],
                reverse=True,
            )[:10]
            internal_ranked = sorted(
                [item for item in split_ranked if item[2] > 0],
                key=lambda item: item[2],
                reverse=True,
            )[:10]

            if external_total > 0 and external_ranked:
                fig.add_trace(
                    go.Bar(
                        x=[domain for domain, *_ in external_ranked],
                        y=[
                            external_count / external_total
                            for _, external_count, _, _ in external_ranked
                        ],
                        marker_color="#00CC96",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            if internal_total > 0 and internal_ranked:
                fig.add_trace(
                    go.Bar(
                        x=[domain for domain, *_ in internal_ranked],
                        y=[
                            internal_count / internal_total
                            for _, _, internal_count, _ in internal_ranked
                        ],
                        marker_color="#E45756",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )
        else:
            if cited_total > 0 and cited_ranked:
                fig.add_trace(
                    go.Bar(
                        x=[domain for domain, _ in cited_ranked],
                        y=[count / cited_total for _, count in cited_ranked],
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        fig.update_layout(
            margin=dict(l=100, b=60, t=30, r=40),
        )
        for row_idx in range(1, len(subplot_titles) + 1):
            fig.update_xaxes(tickangle=-20, automargin=True, row=row_idx, col=1)
            fig.update_yaxes(tickformat=".0%", row=row_idx, col=1, tickfont=dict(size=15))
        fig.add_annotation(
            x=-0.18,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Percentage of URLs",
            textangle=-90,
            showarrow=False,
            font=dict(size=18, color="black"),
        )
        fig = with_paper_style(fig, config=styler(16, 18), legend_pos=None)
        fig.update_layout(margin=dict(l=100, b=60, t=30, r=40))

        summary = {
            "retrieved": [{"domain": d, "count": c} for d, c in retrieved_ranked],
            "cited": [{"domain": d, "count": c} for d, c in cited_ranked],
            "cited_split": [
                {
                    "domain": d,
                    "external_count": ext,
                    "internal_count": intl,
                    "total_count": total,
                }
                for d, ext, intl, total in split_ranked
            ],
        }
        return fig, summary

    rows = _extract_rows_for_models(model_names)
    if common_samples_only:
        rows = _filter_rows_to_common_samples(rows, common_filter_model_names)
        rows = _filter_rows_to_samples_with_web_calls_for_all_models(
            rows, common_filter_model_names
        )
        common_result_keys = {
            row.get("result_key")
            for row in rows
            if row.get("result_key") is not None
        }
        print(
            "Top domains plot (`common_samples`): "
            f"{len(common_result_keys)} samples where all common-filter models called web."
        )
    else:
        rows = _filter_rows_to_common_samples(rows, common_filter_model_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_cols = 3 if separate_cited_external_internal else 2
    column_titles = (
        [
            "Top Search Results Domains",
            "Top Cited Search Results Domains",
            "Top Cited Parametric Domains",
        ]
        if separate_cited_external_internal
        else ["Top Search Results Domains", "Top Cited Domains"]
    )

    subplot_titles = []
    for model_name in model_names:
        display_name = _model_label(model_name)
        for title in column_titles:
            subplot_titles.append(f"{display_name}<br>{title}")

    fig = make_subplots(
        rows=len(model_names),
        cols=num_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    summary = {}
    for row_idx, model_name in enumerate(model_names, start=1):
        model_rows = [row for row in rows if row.get("model") == model_name]
        if not common_samples_only:
            model_rows = [
                row for row in model_rows if bool(row.get("has_web_tool_call", False))
            ]
            print(
                "Top domains plot (`all_web_call_samples`): "
                f"{_model_label(model_name)} uses {len(model_rows)} samples."
            )
        else:
            print(
                "Top domains plot (`common_samples`): "
                f"{_model_label(model_name)} uses {len(model_rows)} samples."
            )
        model_fig, model_summary = _build_model_domain_figure(model_name, model_rows)
        model_slug = str(model_name).replace(".", "-")
        scope_suffix = "common_samples" if common_samples_only else "all_web_call_samples"
        model_file_name = (
            f"replay_top_domains_overall_split_cited__{scope_suffix}__{model_slug}"
            if separate_cited_external_internal
            else f"replay_top_domains_overall__{scope_suffix}__{model_slug}"
        )
        model_fig.write_image(output_dir / f"{model_file_name}.pdf", format="pdf")

        retrieved_ranked = model_summary["retrieved"]
        cited_ranked = model_summary["cited"]
        split_ranked = model_summary["cited_split"]
        retrieved_total = sum(item["count"] for item in retrieved_ranked)
        cited_total = sum(item["count"] for item in cited_ranked)
        external_total = sum(item["external_count"] for item in split_ranked)
        internal_total = sum(item["internal_count"] for item in split_ranked)

        if retrieved_total > 0 and retrieved_ranked:
            fig.add_trace(
                go.Bar(
                    x=[item["domain"] for item in retrieved_ranked],
                    y=[item["count"] / retrieved_total for item in retrieved_ranked],
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

        if separate_cited_external_internal:
            external_ranked = sorted(
                [item for item in split_ranked if item["external_count"] > 0],
                key=lambda item: item["external_count"],
                reverse=True,
            )[:10]
            internal_ranked = sorted(
                [item for item in split_ranked if item["internal_count"] > 0],
                key=lambda item: item["internal_count"],
                reverse=True,
            )[:10]

            if external_total > 0 and external_ranked:
                fig.add_trace(
                    go.Bar(
                        x=[item["domain"] for item in external_ranked],
                        y=[item["external_count"] / external_total for item in external_ranked],
                        marker_color="#00CC96",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=2,
                )

            if internal_total > 0 and internal_ranked:
                fig.add_trace(
                    go.Bar(
                        x=[item["domain"] for item in internal_ranked],
                        y=[item["internal_count"] / internal_total for item in internal_ranked],
                        marker_color="#E45756",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=3,
                )
        else:
            if cited_total > 0 and cited_ranked:
                fig.add_trace(
                    go.Bar(
                        x=[item["domain"] for item in cited_ranked],
                        y=[item["count"] / cited_total for item in cited_ranked],
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=2,
                )

        summary[model_name] = model_summary
        summary[model_name]["num_samples"] = int(len(model_rows))

    scope_suffix = "common_samples" if common_samples_only else "all_web_call_samples"
    combined_retrieved_models = [
        model_name
        for model_name in [
            "gpt-5.3-chat-latest",
            "claude-sonnet-4-6",
            "grok-4.3",
            "deepseek-v4-flash"
        ]
        if model_name in summary
    ]
    if combined_retrieved_models:
        combined_fig = make_subplots(
            rows=len(combined_retrieved_models),
            cols=1,
            subplot_titles=[
                _model_label(model_name)
                for model_name in combined_retrieved_models
            ],
            vertical_spacing=0.27,
        )
        for annotation in combined_fig.layout.annotations:
            annotation.font = dict(size=22, color="black")
        for row_idx, model_name in enumerate(combined_retrieved_models, start=1):
            retrieved_ranked = summary[model_name]["retrieved"][:10]
            retrieved_total = sum(
                item["count"] for item in summary[model_name]["retrieved"]
            )
            if retrieved_total > 0 and retrieved_ranked:
                combined_fig.add_trace(
                    go.Bar(
                        x=[item["domain"] for item in retrieved_ranked],
                        y=[item["count"] / retrieved_total for item in retrieved_ranked],
                        marker=dict(
                            color=platform_color_map.get(model_name, "#636EFA"),
                            pattern=dict(
                                shape=platform_pattern_map.get(model_name, "."),
                                fgcolor="rgba(255,255,255,0.95)",
                                solidity=0.38,
                            ),
                        ),
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=1,
                )
            combined_fig.update_xaxes(
                tickangle=-20,
                automargin=True,
                row=row_idx,
                col=1,
            )
            combined_fig.update_yaxes(
                tickformat=".0%",
                tickfont=dict(size=15),
                nticks=3,
                row=row_idx,
                col=1,
            )

        combined_fig.update_layout(
            margin=dict(l=130, b=60, t=40, r=40),
            height=600
        )
        combined_fig.add_annotation(
            x=-0.23,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Percentage of Search Results URLs",
            textangle=-90,
            showarrow=False,
            font=dict(size=24, color="black"),
        )
        combined_fig = with_paper_style(
            combined_fig, config=styler(18, 18), legend_pos=None
        )
        combined_fig.update_layout(
            margin=dict(l=130, b=60, t=40, r=40),
            xaxis=dict(
                # title=dict(text="Domain"),
                title_standoff=8,
                tickangle=-20,
                automargin=True,
            ),
        )
        combined_file_name = (
            f"replay_top_retrieved_domains_gpt_claude_grok__{scope_suffix}"
        )
        combined_fig.write_image(
            output_dir / f"{combined_file_name}.pdf", format="pdf"
        )

    for row_idx in range(1, len(model_names) + 1):
        for col_idx in range(1, num_cols + 1):
            fig.update_xaxes(tickangle=-20, automargin=True, row=row_idx, col=col_idx)
            fig.update_yaxes(tickformat=".0%", row=row_idx, col=col_idx)

    fig.update_layout(
        margin=dict(l=90, b=60, t=70, r=40),
    )
    fig.add_annotation(
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Percentage of URLs",
        textangle=-90,
        showarrow=False,
        font=dict(size=18, color="black"),
    )

    fig = with_paper_style(fig, config=styler(18, 18), legend_pos=None)
    fig.update_layout(margin=dict(l=90, b=60, t=70, r=40))
    file_name = (
        f"replay_top_domains_overall_split_cited__{scope_suffix}"
        if separate_cited_external_internal
        else f"replay_top_domains_overall__{scope_suffix}"
    )
    fig.write_image(output_dir / f"{file_name}.pdf", format="pdf")

    with open(output_dir / f"{file_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def plot_replay_web_call_topic_distribution_across_platforms(
    model_names=DEFAULT_MODELS,
    output_dir=PLOT_OUTPUT_DIR / "topics",
):
    import plotly.graph_objects as go

    from src.utils.figure_style import with_paper_style, styler

    try:
        import pandas as pd
    except ModuleNotFoundError:
        print("pandas not available; cannot build topic distribution plot.")
        return {}

    rows = _extract_rows_for_models(model_names)
    rows = _filter_rows_to_common_samples(rows, model_names)
    topic_lookup = _build_topic_lookup()
    output_dir.mkdir(parents=True, exist_ok=True)

    platform_frames = []
    summary = {}
    for model_name in model_names:
        model_rows = [
            row
            for row in rows
            if row.get("model") == model_name and bool(row.get("web_queries", []))
        ]

        topic_counts = {}
        for row in model_rows:
            topic = topic_lookup.get(
                (str(row.get("conv_id", "") or ""), str(row.get("turn_id", "") or "")),
                "",
            )
            topic = _normalize_topic_name(topic)
            if not topic or topic.lower() in {"other", "uncategorized", "misc"}:
                continue
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        total = sum(topic_counts.values())
        summary[model_name] = {
            "num_web_call_prompts_with_topic": int(total),
            "topic_counts": dict(sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))),
        }
        if total == 0:
            continue

        for topic, count in topic_counts.items():
            platform_frames.append(
                {
                    "model": model_name,
                    "model_display": _model_label(model_name),
                    "topic": topic,
                    "count": int(count),
                    "rate": float(count / total),
                }
            )

    if not platform_frames:
        print("No topic rows found for replay web-call prompts.")
        return summary

    rates_df = pd.DataFrame(platform_frames)
    base_model = "gpt-5.3-chat-latest" if "gpt-5.3-chat-latest" in model_names else model_names[0]
    base_rates = (
        rates_df[rates_df["model"] == base_model]
        .set_index("topic")["rate"]
    )
    base_sorted_topics = base_rates.sort_values(ascending=False).index.tolist()
    other_topics = sorted(
        topic for topic in rates_df["topic"].unique().tolist() if topic not in base_sorted_topics
    )
    topic_order = base_sorted_topics + other_topics

    fig = go.Figure()
    for model_name in model_names:
        model_df = rates_df[rates_df["model"] == model_name].copy()
        if len(model_df) == 0:
            continue
        model_df["topic"] = pd.Categorical(
            model_df["topic"],
            categories=topic_order,
            ordered=True,
        )
        model_df = model_df.sort_values("topic")
        fig.add_trace(
            go.Bar(
                x=model_df["topic"].astype(str),
                y=model_df["rate"],
                name=_model_label(model_name),
                hovertemplate=(
                    "Topic: %{x}<br>"
                    "Share of web-call prompts: %{y:.1%}<br>"
                    "Count: %{customdata}<extra></extra>"
                ),
                customdata=model_df["count"],
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Topic",
        yaxis_title="Share of Web-Call Prompts",
        margin=dict(l=70, b=110, t=30, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(tickangle=-35, categoryorder="array", categoryarray=topic_order)
    fig.update_yaxes(tickformat=".0%")

    fig = with_paper_style(fig, config=styler(18, 16))
    fig.update_layout(
        margin=dict(l=70, b=110, t=30, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.write_image(
        output_dir / "replay_web_call_topic_distribution_across_platforms.pdf",
        format="pdf",
    )

    with open(
        output_dir / "replay_web_call_topic_distribution_across_platforms_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _build_replay_query_specificity_stage_df(df, max_web_stage_bucket=3):
    from collections import Counter

    specificity_dimensions = ["temporal", "geographic", "entity", "numeric"]
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

    def _add_score(model_name, stage_idx, dimension, score):
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
        stage_dimension_score_counts.setdefault(model_name, {})
        stage_dimension_score_counts[model_name].setdefault(stage_idx, {})
        stage_dimension_score_counts[model_name][stage_idx].setdefault(
            dimension, Counter()
        )
        stage_dimension_score_counts[model_name][stage_idx][dimension][score] += 1

    for _, row in df.iterrows():
        model_name = row.get("model")
        if not model_name:
            continue
        user_query_specificity = _safe_json_value(
            row.get("user_query_specificity", "{}"),
            {},
        )
        if isinstance(user_query_specificity, dict):
            for dimension, judgment in user_query_specificity.items():
                if isinstance(judgment, dict):
                    _add_score(model_name, 0, dimension, judgment.get("score"))

        web_query_specificity_info = _safe_json_value(
            row.get("web_query_specificity_info", "[]"),
            [],
        )
        if not isinstance(web_query_specificity_info, list):
            continue
        for item in web_query_specificity_info:
            if not isinstance(item, dict):
                continue
            try:
                iteration_idx = int(item.get("iteration"))
            except (TypeError, ValueError):
                continue
            specificity = item.get("specificity", {})
            if not isinstance(specificity, dict):
                continue
            for dimension, judgment in specificity.items():
                if isinstance(judgment, dict):
                    _add_score(model_name, iteration_idx, dimension, judgment.get("score"))

    rows = []
    for model_name in sorted(stage_dimension_score_counts):
        for stage_idx in sorted(stage_dimension_score_counts[model_name]):
            if stage_idx == 0:
                stage_label = "User"
            elif stage_idx == max_web_stage_bucket:
                stage_label = f"Iter. {max_web_stage_bucket}+"
            else:
                stage_label = f"Iter. {stage_idx}"
            for dimension in specificity_dimensions:
                score_counter = (
                    stage_dimension_score_counts[model_name]
                    .get(stage_idx, {})
                    .get(dimension, Counter())
                )
                total = sum(score_counter.values())
                for score in score_order:
                    rows.append(
                        {
                            "model": model_name,
                            "model_display": _model_label(model_name),
                            "stage_idx": stage_idx,
                            "stage_label": stage_label,
                            "dimension": dimension,
                            "dimension_display": specificity_label_map[dimension],
                            "score": score,
                            "count": score_counter.get(score, 0),
                            "total": total,
                            "rate": (score_counter.get(score, 0) / total) if total else 0.0,
                        }
                    )
    return rows


def query_specificity_evaluation(
    model_names=DEFAULT_MODELS,
    evaluator_model="gpt-4o-mini",
    output_stem="replay_query_specificity",
):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    client = OpenAI(api_key=api_key)
    replay_rows = _build_replay_query_eval_rows(model_names)
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

    specificity_cache = {}
    user_query_specificity_by_result_key = {}

    def _evaluate_query_specificity(query_text):
        query_text = str(query_text or "").strip()
        if not query_text:
            return {}
        if query_text in specificity_cache:
            return specificity_cache[query_text]

        specificity = {}
        for dimension, (system_prompt, user_prompt_template) in specificity_dimensions.items():
            eval_result = _run_judge(
                client=client,
                model_name=evaluator_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt_template.format(QUERY=query_text),
            )
            parsed = eval_result["parsed_judgment"]
            if not isinstance(parsed, dict):
                parsed = {}
            try:
                score = int(parsed.get("score"))
            except (TypeError, ValueError):
                score = None
            specificity[dimension] = {
                "score": score,
                "reason": str(parsed.get("reason", "")).strip(),
            }
        specificity_cache[query_text] = specificity
        return specificity

    records = []
    for row in tqdm(replay_rows, total=len(replay_rows)):
        user_query = str(row.get("user_prompt") or "").strip()
        if not user_query:
            continue
        result_key = row.get("result_key")
        try:
            if result_key in user_query_specificity_by_result_key:
                user_query_specificity = user_query_specificity_by_result_key[result_key]
            else:
                user_query_specificity = _evaluate_query_specificity(user_query)
                user_query_specificity_by_result_key[result_key] = user_query_specificity
        except Exception as exc:
            print("query_specificity user_query", row.get("model"), row.get("conv_id"), row.get("turn_id"), exc)
            continue

        web_query_specificity_info = []
        for iteration_idx, query_group in enumerate(row.get("web_queries", []), start=1):
            if not isinstance(query_group, list):
                query_group = [query_group]
            for web_query in query_group:
                if not isinstance(web_query, str) or not web_query.strip():
                    continue
                try:
                    web_query_specificity_info.append(
                        {
                            "query": web_query.strip(),
                            "iteration": iteration_idx,
                            "specificity": _evaluate_query_specificity(web_query.strip()),
                        }
                    )
                except Exception as exc:
                    print("query_specificity web_query", row.get("model"), row.get("conv_id"), row.get("turn_id"), exc)

        records.append(
            {
                "model": row.get("model"),
                "provider": row.get("provider"),
                "result_key": row.get("result_key"),
                "sample_idx": row.get("sample_idx"),
                "sample_source": row.get("sample_source"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "user_prompt": user_query,
                "web_queries": row.get("web_queries", []),
                "user_query_specificity": user_query_specificity,
                "web_query_specificity_info": web_query_specificity_info,
            }
        )
        if len(records) % 10 == 0:
            _save_replay_query_eval_records(records, output_stem)

    return _save_replay_query_eval_records(records, output_stem)


def plot_query_specificity_distribution_by_iteration(
    input_stem="replay_query_specificity",
    output_dir=PLOT_OUTPUT_DIR / "query_reformulations",
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from src.utils.figure_style import with_paper_style, styler

    df = _load_replay_query_eval_df(input_stem)
    if df is None or df.empty:
        print("No replay query specificity rows to plot.")
        return {}

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
        user_specificity_dict, query_specificity_dict, dimension
    ):
        user_scores = _specificity_score_vector(user_specificity_dict)
        query_scores = _specificity_score_vector(query_specificity_dict)
        dim_key = str(dimension).strip().lower()
        if dim_key not in user_scores or dim_key not in query_scores:
            return None
        delta = query_scores[dim_key] - user_scores[dim_key]
        if delta > 0:
            return 1
        if delta == 0:
            return 0
        return 0

    plot_rows = _build_replay_query_specificity_stage_df(df)
    if not plot_rows:
        print("No replay query specificity stage rows to plot.")
        return {}

    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_csv(REPLAY_QUERY_EVAL_OUTPUT_DIR / f"{input_stem}_distribution_by_iteration.csv", index=False)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    overall_stage_rows = []
    dimension_direction_stage_rows = []

    summary = {}
    for model_name in sorted(plot_df["model"].unique()):
        model_df = plot_df[plot_df["model"] == model_name].copy()
        raw_model_df = df[df["model"] == model_name].copy()
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_yaxes=True,
            subplot_titles=dimension_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.25,
        )
        fig.update_annotations(font_size=22)
        for dim_idx, dimension in enumerate(dimensions):
            row_idx = (dim_idx // 2) + 1
            col_idx = (dim_idx % 2) + 1
            dimension_df = model_df[model_df["dimension"] == dimension].copy()
            stage_labels = (
                dimension_df[["stage_idx", "stage_label"]]
                .drop_duplicates()
                .sort_values("stage_idx")["stage_label"]
                .tolist()
            )
            for score in score_order:
                score_df = dimension_df[dimension_df["score"] == score].sort_values("stage_idx")
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
                        hovertemplate="Stage: %{x}<br>Score: %{fullData.name}<br>Share: %{y:.1%}<br>Count: %{customdata[0]} / %{customdata[1]}<extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            fig.update_xaxes(categoryorder="array", categoryarray=stage_labels, row=row_idx, col=col_idx, tickfont=dict(size=18),)

        fig.update_layout(barmode="stack", margin=dict(t=90, b=100, l=100, r=30), legend_title="")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.0])
        fig.add_annotation(x=-0.18, y=0.5, xref="paper", yref="paper", text="Share", showarrow=False, textangle=-90, font=dict(size=20))
        fig.add_annotation(x=0.5, y=-0.25, xref="paper", yref="paper", text="Query Formulation Iteration", showarrow=False, font=dict(size=20))
        file_name = f"replay_query_specificity_distribution_by_iteration__{model_name}"
        fig.write_html(output_dir / f"{file_name}.html")
        fig = with_paper_style(fig, config=styler(20, 17), legend_pos=(1, 1.25))
        fig.write_image(output_dir / f"{file_name}.pdf", format="pdf")
        summary[model_name] = {
            "dimension_distribution_rows": model_df.to_dict(orient="records"),
        }

        stage_direction_values = {}
        dimension_stage_direction_values = {
            dimension: {} for dimension in dimensions
        }
        for _, raw_row in raw_model_df.iterrows():
            user_specificity = _safe_json_value(
                raw_row.get("user_query_specificity", {}),
                {},
            )

            web_query_specificity_info = _safe_json_value(
                raw_row.get("web_query_specificity_info", []),
                [],
            )
            if not isinstance(web_query_specificity_info, list):
                continue
            for item in web_query_specificity_info:
                if not isinstance(item, dict):
                    continue
                try:
                    iteration_idx = int(item.get("iteration"))
                except (TypeError, ValueError):
                    continue
                stage_idx = min(iteration_idx, 3) if iteration_idx > 0 else 0
                direction = _overall_specificity_direction(
                    user_specificity,
                    item.get("specificity", {}),
                )
                if direction is not None:
                    stage_direction_values.setdefault(stage_idx, []).append(direction)
                for dimension in dimensions:
                    dimension_direction = _dimension_specificity_direction(
                        user_specificity,
                        item.get("specificity", {}),
                        dimension,
                    )
                    if dimension_direction is not None:
                        dimension_stage_direction_values[dimension].setdefault(
                            stage_idx, []
                        ).append(dimension_direction)

        overall_rows = []
        for stage_idx in sorted(stage_direction_values):
            if stage_idx >= 3:
                stage_label = "Iter. 3+"
            else:
                stage_label = f"Iter. {stage_idx}"
            values = stage_direction_values[stage_idx]
            mean_value = sum(values) / len(values) if values else None
            row = {
                "model": model_name,
                "model_display": _model_label(model_name),
                "stage_idx": int(stage_idx),
                "stage_label": stage_label,
                "mean_overall_specificity_direction": float(mean_value) if mean_value is not None else None,
                "percentage_overall_specificity_direction": float(mean_value * 100.0) if mean_value is not None else None,
                "count": int(len(values)),
            }
            overall_rows.append(row)
            overall_stage_rows.append(row)
        summary[model_name]["overall_specificity_rows"] = overall_rows

        dimension_rows = []
        for dimension in dimensions:
            for stage_idx in sorted(dimension_stage_direction_values[dimension]):
                if stage_idx >= 3:
                    stage_label = "Iter. 3+"
                else:
                    stage_label = f"Iter. {stage_idx}"
                values = dimension_stage_direction_values[dimension][stage_idx]
                mean_value = sum(values) / len(values) if values else None
                row = {
                    "model": model_name,
                    "model_display": _model_label(model_name),
                    "dimension": dimension,
                    "dimension_display": dimension.title(),
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
                dimension_rows.append(row)
                dimension_direction_stage_rows.append(row)
        summary[model_name]["dimension_direction_rows"] = dimension_rows

    with open(output_dir / "replay_query_specificity_distribution_by_iteration_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if overall_stage_rows:
        overall_df = pd.DataFrame(overall_stage_rows)
        overall_df.to_csv(
            output_dir / "replay_query_specificity_overall_by_iteration.csv",
            index=False,
        )
        line_fig = go.Figure()
        color_sequence = {
            "gpt-5.3-chat-latest": "#636EFA",
            "claude-sonnet-4-6": "#EF553B",
            "grok-4.3": "#00CC96",
            "deepseek-v4-flash": "#AB63FA",
        }
        marker_sequence = {
            "gpt-5.3-chat-latest": "circle",
            "claude-sonnet-4-6": "star",
            "grok-4.3": "x",
            "deepseek-v4-flash": "diamond",
        }
        ordered_models = [
            model_name
            for model_name in [
                "gpt-5.3-chat-latest",
                "claude-sonnet-4-6",
                "grok-4.3",
                "deepseek-v4-flash",
            ]
            if model_name in set(overall_df["model"].unique())
        ]
        for model_name in ordered_models:
            model_line_df = overall_df[overall_df["model"] == model_name].sort_values("stage_idx")
            line_fig.add_trace(
                go.Scatter(
                    x=model_line_df["stage_idx"],
                    y=model_line_df["percentage_overall_specificity_direction"],
                    mode="lines+markers",
                    name=_model_label(model_name),
                    line=dict(color=color_sequence.get(model_name), width=4),
                    marker=dict(
                        size=16,
                        color=color_sequence.get(model_name),
                        symbol=marker_sequence.get(model_name, "circle"),
                    ),
                    customdata=model_line_df[["count", "stage_label"]].values,
                    hovertemplate=(
                        "Model: %{fullData.name}<br>"
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
            ticktext=["User → Iter. 1", "Iter. 1 → Iter. 2", "Iter. 2+ → Iter. 3+"],
            range=[0.8, 3.4],
            tickfont=dict(size=21)
            
        )
        line_fig.update_yaxes(ticksuffix="%")
        line_file_name = "replay_query_specificity_overall_by_iteration"
        line_fig.write_html(output_dir / f"{line_file_name}.html")
        line_fig = with_paper_style(line_fig, config=styler(24, 24), legend_pos=(0.9, 1.2))
        line_fig.write_image(output_dir / f"{line_file_name}.pdf", format="pdf")

    if dimension_direction_stage_rows:
        dimension_direction_df = pd.DataFrame(dimension_direction_stage_rows)
        dimension_direction_df.to_csv(
            output_dir / "replay_query_specificity_dimension_direction_by_iteration.csv",
            index=False,
        )
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
        for model_name in sorted(dimension_direction_df["model"].unique()):
            model_dimension_df = dimension_direction_df[
                dimension_direction_df["model"] == model_name
            ].copy()
            model_overall_df = pd.DataFrame(
                summary.get(model_name, {}).get("overall_specificity_rows", [])
            )
            per_model_line_fig = go.Figure()
            for dimension in dimensions:
                if dimension == "numeric":
                    continue
                dimension_df = model_dimension_df[
                    model_dimension_df["dimension"] == dimension
                ].sort_values("stage_idx")
                if dimension_df.empty:
                    continue
                per_model_line_fig.add_trace(
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
                        marker=dict(size=12, color=dimension_line_color_map[dimension]),
                        customdata=dimension_df[["count", "stage_label"]].values,
                        hovertemplate=(
                            "Dimension: %{fullData.name}<br>"
                            "Stage: %{customdata[1]}<br>"
                            "Mean direction: %{y:.1f}%<br>"
                            "Queries: %{customdata[0]}<extra></extra>"
                        ),
                    )
                )
            # if not model_overall_df.empty:
            #     model_overall_df = model_overall_df.sort_values("stage_idx")
            #     per_model_line_fig.add_trace(
            #         go.Scatter(
            #             x=model_overall_df["stage_idx"],
            #             y=model_overall_df["percentage_overall_specificity_direction"],
            #             mode="lines+markers",
            #             name=dimension_display_map["overall"],
            #             line=dict(
            #                 color=dimension_line_color_map["overall"],
            #                 dash=dimension_line_dash_map["overall"],
            #                 width=4,
            #             ),
            #             marker=dict(
            #                 size=12, color=dimension_line_color_map["overall"]
            #             ),
            #             customdata=model_overall_df[["count", "stage_label"]].values,
            #             hovertemplate=(
            #                 "Dimension: %{fullData.name}<br>"
            #                 "Stage: %{customdata[1]}<br>"
            #                 "Mean direction: %{y:.1f}%<br>"
            #                 "Queries: %{customdata[0]}<extra></extra>"
            #             ),
            #         )
            #     )
            per_model_line_fig.update_layout(
                title=f"{_model_label(model_name)}: specificity direction by dimension",
                xaxis_title="Query Formulation Iteration",
                yaxis_title="Avg Specificity Increase (%)",
                margin=dict(t=50, b=80, l=80, r=30),
                legend_title="",
                legend=dict(font=dict(size=20)),
            )
            per_model_line_fig.update_xaxes(
                tickmode="array",
                tickvals=[1, 2, 3],
                ticktext=["User → Iter. 1", "Iter. 1 → Iter. 2", "Iter. 2+ → Iter. 3+"],
                range=[0.8, 3.4],
                # tickfont=dict(size=21),
            )
            per_model_line_fig.update_yaxes(ticksuffix="%")
            per_model_file_name = (
                f"replay_query_specificity_dimension_direction_by_iteration__{model_name}"
            )
            per_model_line_fig.write_html(output_dir / f"{per_model_file_name}.html")
            per_model_line_fig = with_paper_style(
                per_model_line_fig,
                config=styler(24, 24),
                # legend_pos=(0.9, 1.18),
            )
            per_model_line_fig.write_image(
                output_dir / f"{per_model_file_name}.pdf",
                format="pdf",
            )

    return summary


def reasons_for_another_web_query(
    model_names=DEFAULT_MODELS,
    evaluator_model="gpt-4o-mini",
    output_stem="replay_web_query_transition_reasons",
):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    client = OpenAI(api_key=api_key)
    replay_rows = [
        row
        for row in _build_replay_query_eval_rows(model_names)
        if any(isinstance(group, list) and any(isinstance(q, str) and q.strip() for q in group) for group in row.get("web_queries", []))
    ]

    records = []
    for row in tqdm(replay_rows, total=len(replay_rows)):
        user_query = str(row.get("user_prompt") or "").strip()
        if not user_query:
            continue

        structured_web_queries = []
        for loop_idx, query_group in enumerate(row.get("web_queries", []), start=1):
            if not isinstance(query_group, list):
                query_group = [query_group]
            cleaned_queries = [str(q).strip() for q in query_group if isinstance(q, str) and q.strip()]
            if cleaned_queries:
                structured_web_queries.append({"loop_idx": loop_idx, "queries": cleaned_queries})
        if not structured_web_queries:
            continue

        loop_query_records = {}
        structured_query_records = []
        structured_thinking_records = []
        for loop_entry in structured_web_queries:
            loop_records = []
            for query_idx, query in enumerate(loop_entry["queries"], start=1):
                record = {
                    "query_id": f"{loop_entry['loop_idx']}.{query_idx}",
                    "loop_idx": loop_entry["loop_idx"],
                    "query_idx": query_idx,
                    "query": query,
                }
                loop_records.append(record)
                structured_query_records.append(record)
                structured_thinking_records.append(
                    {
                        "query_id": f"{loop_entry['loop_idx']}.{query_idx}",
                        "thinking_trace": "",
                    }
                )
            loop_query_records[loop_entry["loop_idx"]] = loop_records

        transition_candidates = []
        first_loop_idx = structured_web_queries[0]["loop_idx"]
        for to_record in loop_query_records.get(first_loop_idx, []):
            transition_candidates.append(
                {
                    "from": "U",
                    "to": to_record["query_id"],
                    "from_loop_idx": 0,
                    "to_loop_idx": first_loop_idx,
                    "transition_kind": "user_to_first_web_turn",
                }
            )
        for loop_pos in range(len(structured_web_queries) - 1):
            from_loop_idx = structured_web_queries[loop_pos]["loop_idx"]
            to_loop_idx = structured_web_queries[loop_pos + 1]["loop_idx"]
            for from_record in loop_query_records.get(from_loop_idx, []):
                for to_record in loop_query_records.get(to_loop_idx, []):
                    transition_candidates.append(
                        {
                            "from": from_record["query_id"],
                            "to": to_record["query_id"],
                            "from_loop_idx": from_loop_idx,
                            "to_loop_idx": to_loop_idx,
                            "transition_kind": "web_turn_to_web_turn",
                        }
                    )
        if not transition_candidates:
            continue

        web_queries_text = "\n".join(f"({item['query_id']}) {item['query']}" for item in structured_query_records)
        transition_candidates_text = "\n".join(f"({item['from']} -> {item['to']})" for item in transition_candidates)
        thinking_trace_lines = [
            f"({item['query_id']}) {item['thinking_trace']}"
            for item in structured_thinking_records
            if item["thinking_trace"]
        ]
        thinking_traces_text = "\n".join(thinking_trace_lines)

        try:
            reason_eval = _run_judge(
                client=client,
                model_name=evaluator_model,
                system_prompt=SYSTEM_PROMPT_QUERY_REASON,
                user_prompt=USER_PROMPT_QUERY_REASON.format(
                    user_query=user_query,
                    web_queries=web_queries_text,
                    thinking_traces=thinking_traces_text,
                    transition_candidates=transition_candidates_text,
                ),
            )
        except Exception as exc:
            print("reasons_for_another_web_query reason", row.get("model"), row.get("conv_id"), row.get("turn_id"), exc)
            continue

        reason_parsed = reason_eval["parsed_judgment"]
        transitions = reason_parsed.get("transitions", []) if isinstance(reason_parsed, dict) else []
        valid_transition_pairs = {(item["from"], item["to"]) for item in transition_candidates}
        normalized_reason_transitions = []
        reason_transition_by_pair = {}
        for transition in transitions:
            if not isinstance(transition, dict):
                continue
            from_id = _normalize_reason_transition_endpoint(transition.get("from"))
            to_id = _normalize_reason_transition_endpoint(transition.get("to"))
            pair = (from_id, to_id)
            if (
                not from_id
                or not to_id
                or pair not in valid_transition_pairs
                or pair in reason_transition_by_pair
            ):
                continue
            normalized_transition = {
                "from": from_id,
                "to": to_id,
                "label": _normalize_query_reason_label(transition.get("label", "")),
                "reasoning": str(transition.get("reasoning", "")).strip(),
            }
            reason_transition_by_pair[pair] = normalized_transition
            normalized_reason_transitions.append(normalized_transition)

        validation_transition_candidates = []
        for transition in transition_candidates:
            validation_transition_candidates.append(
                {
                    "from": transition["to"],
                    "to": transition["from"],
                    "from_query": transition.get("to_query"),
                    "to_query": transition.get("from_query"),
                    "from_loop_idx": transition["to_loop_idx"],
                    "to_loop_idx": transition["from_loop_idx"],
                    "transition_kind": transition["transition_kind"],
                }
            )
        validation_transition_candidates_text = "\n".join(
            f"({item['from']} -> {item['to']})"
            for item in validation_transition_candidates
        )
        valid_validation_transition_pairs = {
            (item["from"], item["to"]) for item in validation_transition_candidates
        }

        try:
            validation_eval = _run_judge(
                client=client,
                model_name=evaluator_model,
                system_prompt=SYSTEM_PROMPT_QUERY_REASON_VALIDATOR,
                user_prompt=USER_PROMPT_QUERY_REASON_VALIDATOR.format(
                    user_query=user_query,
                    web_queries=web_queries_text,
                    transition_candidates=validation_transition_candidates_text,
                ),
            )
        except Exception as exc:
            print("reasons_for_another_web_query validate", row.get("model"), row.get("conv_id"), row.get("turn_id"), exc)
            continue

        validator_parsed = validation_eval["parsed_judgment"]
        validator_transitions = (
            validator_parsed.get("transitions", [])
            if isinstance(validator_parsed, dict)
            else []
        )
        validator_transition_by_pair = {}
        normalized_validator_transitions = []
        for transition in validator_transitions:
            if not isinstance(transition, dict):
                continue
            from_id = _normalize_reason_transition_endpoint(transition.get("from"))
            to_id = _normalize_reason_transition_endpoint(transition.get("to"))
            pair = (from_id, to_id)
            if (
                not from_id
                or not to_id
                or pair not in valid_validation_transition_pairs
                or pair in validator_transition_by_pair
            ):
                continue
            normalized_transition = {
                "from": from_id,
                "to": to_id,
                "label": _normalize_query_reason_label(transition.get("label", "")),
                "reasoning": str(transition.get("reasoning", "")).strip(),
            }
            validator_transition_by_pair[pair] = normalized_transition
            normalized_validator_transitions.append(normalized_transition)

        records.append(
            {
                "model": row.get("model"),
                "provider": row.get("provider"),
                "result_key": row.get("result_key"),
                "sample_idx": row.get("sample_idx"),
                "sample_source": row.get("sample_source"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "user_query": user_query,
                "web_queries": structured_query_records,
                "web_queries_structured_text": web_queries_text,
                "thinking_traces": structured_thinking_records,
                "transition_candidates": transition_candidates,
                "transition_candidates_text": transition_candidates_text,
                "validator_transition_candidates": validation_transition_candidates,
                "validator_transition_candidates_text": validation_transition_candidates_text,
                "query_reason_parsed_judgment_judgment": reason_eval["parsed_judgment"],
                "query_reason_validator_parsed_judgment_judgment": validation_eval["parsed_judgment"],
                "query_reason_transitions_normalized": normalized_reason_transitions,
                "query_reason_validator_transitions_normalized": normalized_validator_transitions,
            }
        )
        if len(records) % 10 == 0:
            _save_replay_query_eval_records(records, output_stem)

    return _save_replay_query_eval_records(records, output_stem)


def plot_reasons_for_another_web_query_distribution_all_models(
    input_stem="replay_web_query_transition_reasons",
    model_names=DEFAULT_MODELS,
    output_file_name="replay_reasons_for_another_web_query_distribution_all_models",
    output_dir=PLOT_OUTPUT_DIR / "query_reformulations",
):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from src.utils.figure_style import with_paper_style, styler

    df = _load_replay_query_eval_df(input_stem)
    if df is None or df.empty:
        print("No replay query-transition reason data found.")
        return None

    reason_order = ["Query Rewriting", "Query Expansion", "Hybrid", "Other"]
    color_map = {
        "Query Rewriting": "#1f77b4",
        "Query Expansion": "#2ca02c",
        "Hybrid": "#ff7f0e",
        "Other": "#b59b00",
    }
    symbol_map = {
        "Query Rewriting": "circle",
        "Query Expansion": "square",
        "Hybrid": "diamond",
        "Other": "x",
    }

    def _as_dict(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(value)
                except Exception:
                    return {}
        return {}

    def _coerce_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_loop_idx_from_endpoint(endpoint):
        normalized = _normalize_reason_transition_endpoint(endpoint)
        if not normalized or normalized == "U":
            return None
        try:
            return int(str(normalized).split(".", 1)[0])
        except (TypeError, ValueError):
            return None

    aggregate_from_iteration = 3

    def _bucket_iteration(iteration_idx):
        if iteration_idx >= aggregate_from_iteration:
            return aggregate_from_iteration
        return iteration_idx

    def _transition_group_label(iteration_idx, use_arrow=False):
        arrow = " → " if use_arrow else " -> "
        if iteration_idx == 1:
            return f"User{arrow}Iter. 1"
        if iteration_idx >= aggregate_from_iteration:
            return f"Iter. {aggregate_from_iteration-1}+{arrow}Iter. {aggregate_from_iteration}+"
        return f"Iter. {iteration_idx - 1}{arrow}Iter. {iteration_idx}"

    def _build_transition_meta_by_pair(transitions):
        transition_meta = {}
        for transition in transitions:
            if not isinstance(transition, dict):
                continue
            transition_key = (
                _normalize_reason_transition_endpoint(transition.get("from")),
                _normalize_reason_transition_endpoint(transition.get("to")),
            )
            if not transition_key[0] or not transition_key[1]:
                continue
            transition_meta[transition_key] = {
                "transition_kind": str(transition.get("transition_kind", "")).strip(),
                "from_loop_idx": _coerce_int(transition.get("from_loop_idx")),
                "to_loop_idx": _coerce_int(transition.get("to_loop_idx")),
            }
        return transition_meta

    def _infer_iteration_idx(transition_key, transition_meta, flipped=False):
        transition_kind = str(transition_meta.get("transition_kind", "")).strip()
        from_loop_idx = transition_meta.get("from_loop_idx")
        to_loop_idx = transition_meta.get("to_loop_idx")
        if from_loop_idx is None:
            from_loop_idx = _parse_loop_idx_from_endpoint(transition_key[0])
        if to_loop_idx is None:
            to_loop_idx = _parse_loop_idx_from_endpoint(transition_key[1])
        if flipped:
            if transition_key[1] == "U" or transition_kind == "user_to_first_web_turn":
                return 1
            if (
                from_loop_idx is not None
                and to_loop_idx is not None
                and from_loop_idx == to_loop_idx + 1
            ):
                return from_loop_idx
            return None
        if transition_key[0] == "U" or transition_kind == "user_to_first_web_turn":
            return 1
        if (
            from_loop_idx is not None
            and to_loop_idx is not None
            and to_loop_idx == from_loop_idx + 1
        ):
            return to_loop_idx
        return None

    def _aggregate_labels_for_destination_query(labels):
        normalized_labels = [label for label in labels if label in reason_order]
        if not normalized_labels:
            return ""
        label_set = set(normalized_labels)
        if "Hybrid" in label_set:
            return "Hybrid"
        if "Query Rewriting" in label_set and "Query Expansion" in label_set:
            return "Hybrid"
        non_other_labels = label_set - {"Other"}
        if non_other_labels == {"Query Rewriting"}:
            return "Query Rewriting"
        if non_other_labels == {"Query Expansion"}:
            return "Query Expansion"
        if not non_other_labels:
            return "Other"
        if len(non_other_labels) == 1:
            return next(iter(non_other_labels))
        return "Hybrid"

    plot_rows = []
    for model_name in model_names:
        model_df = df[df["model"] == model_name].copy()
        query_iteration_totals_before = {}
        query_iteration_reason_counts_before = {}
        for _, row in model_df.iterrows():
            normalized_reason_transitions = _safe_json_value(
                row.get("query_reason_transitions_normalized", []),
                [],
            )
            normalized_validator_transitions = _safe_json_value(
                row.get("query_reason_validator_transitions_normalized", []),
                [],
            )
            transition_candidates = _safe_json_value(
                row.get("transition_candidates", []),
                [],
            )
            validator_transition_candidates = _safe_json_value(
                row.get("validator_transition_candidates", []),
                [],
            )
            if not isinstance(normalized_reason_transitions, list):
                normalized_reason_transitions = []
            if not isinstance(normalized_validator_transitions, list):
                normalized_validator_transitions = []
            if not isinstance(transition_candidates, list):
                transition_candidates = []
            if not isinstance(validator_transition_candidates, list):
                validator_transition_candidates = []
            if not validator_transition_candidates and transition_candidates:
                for transition in transition_candidates:
                    if not isinstance(transition, dict):
                        continue
                    validator_transition_candidates.append(
                        {
                            "from": transition.get("to"),
                            "to": transition.get("from"),
                            "from_loop_idx": transition.get("to_loop_idx"),
                            "to_loop_idx": transition.get("from_loop_idx"),
                            "transition_kind": transition.get("transition_kind", ""),
                        }
                    )

            reason_judgment = _as_dict(
                row.get("query_reason_parsed_judgment_judgment", {})
            )
            validator_judgment = _as_dict(
                row.get("query_reason_validator_parsed_judgment_judgment", {})
            )
            original_transitions = normalized_reason_transitions or (
                reason_judgment.get("transitions", [])
                if isinstance(reason_judgment, dict)
                else []
            )
            validator_transitions = normalized_validator_transitions or (
                validator_judgment.get("transitions", [])
                if isinstance(validator_judgment, dict)
                else []
            )

            reason_by_pair = {}
            for transition in original_transitions:
                if not isinstance(transition, dict):
                    continue
                transition_key = (
                    _normalize_reason_transition_endpoint(transition.get("from")),
                    _normalize_reason_transition_endpoint(transition.get("to")),
                )
                if transition_key[0] and transition_key[1]:
                    reason_by_pair[transition_key] = transition

            validator_by_pair = {}
            for transition in validator_transitions:
                if not isinstance(transition, dict):
                    continue
                transition_key = (
                    _normalize_reason_transition_endpoint(transition.get("from")),
                    _normalize_reason_transition_endpoint(transition.get("to")),
                )
                if transition_key[0] and transition_key[1]:
                    validator_by_pair[transition_key] = {
                        "label": _normalize_query_reason_label(transition.get("label", "")),
                        "reasoning": str(transition.get("reasoning", "")).strip(),
                    }

            transition_meta_by_pair = _build_transition_meta_by_pair(transition_candidates)
            validator_transition_meta_by_pair = _build_transition_meta_by_pair(
                validator_transition_candidates
            )
            transition_keys_before = list(transition_meta_by_pair.keys()) or list(
                reason_by_pair.keys()
            )

            incoming_labels_by_destination_query = {}
            destination_query_iteration_bucket = {}
            for transition_key in transition_keys_before:
                original_label = _normalize_query_reason_label(
                    reason_by_pair.get(transition_key, {}).get("label", "")
                )
                iteration_idx = _infer_iteration_idx(
                    transition_key,
                    transition_meta_by_pair.get(transition_key, {}),
                    flipped=False,
                )
                if iteration_idx is None:
                    continue
                iteration_bucket = _bucket_iteration(iteration_idx)
                if original_label in reason_order:
                    destination_query = transition_key[1]
                    if destination_query:
                        incoming_labels_by_destination_query.setdefault(
                            destination_query,
                            [],
                        ).append(original_label)
                        destination_query_iteration_bucket.setdefault(
                            destination_query,
                            iteration_bucket,
                        )

            for destination_query, incoming_labels in incoming_labels_by_destination_query.items():
                aggregate_label = _aggregate_labels_for_destination_query(incoming_labels)
                if aggregate_label not in reason_order:
                    continue
                iteration_bucket = destination_query_iteration_bucket.get(destination_query)
                if iteration_bucket is None:
                    continue
                query_iteration_totals_before[iteration_bucket] = (
                    query_iteration_totals_before.get(iteration_bucket, 0) + 1
                )
                key = (iteration_bucket, aggregate_label)
                query_iteration_reason_counts_before[key] = (
                    query_iteration_reason_counts_before.get(key, 0) + 1
                )

            transition_keys_after = list(validator_transition_meta_by_pair.keys()) or list(
                validator_by_pair.keys()
            )
            for transition_key in transition_keys_after:
                _ = _infer_iteration_idx(
                    transition_key,
                    validator_transition_meta_by_pair.get(transition_key, {}),
                    flipped=True,
                )

        for iteration_idx in sorted(query_iteration_totals_before):
            total = query_iteration_totals_before[iteration_idx]
            for reason in reason_order:
                count = query_iteration_reason_counts_before.get((iteration_idx, reason), 0)
                plot_rows.append(
                    {
                        "model": model_name,
                        "model_display": _model_label(model_name),
                        "iteration": iteration_idx,
                        "transition_group": _transition_group_label(
                            iteration_idx,
                            use_arrow=False,
                        ),
                        "reason": reason,
                        "count": count,
                        "total": total,
                        "rate": (count / total) if total else 0.0,
                    }
                )

    if not plot_rows:
        print("No replay query-reason rate rows to plot.")
        return None

    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_csv(REPLAY_QUERY_EVAL_OUTPUT_DIR / f"{output_file_name}.csv", index=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotted_models = [m for m in model_names if m in set(plot_df["model"])]
    fig = make_subplots(
        rows=1,
        cols=len(plotted_models),
        shared_yaxes=True,
        subplot_titles=[_model_label(m) for m in plotted_models],
        horizontal_spacing=0.04 if len(plotted_models) > 1 else 0.02,
    )
    fig.update_annotations(font_size=24)
    y_max = 0.0
    for col_idx, model_name in enumerate(plotted_models, start=1):
        model_plot_df = plot_df[plot_df["model"] == model_name].copy()
        y_max = max(y_max, float(model_plot_df["rate"].max()))
        iteration_values = sorted(model_plot_df["iteration"].unique())
        ticktext = (
            model_plot_df[["iteration", "transition_group"]]
            .drop_duplicates()
            .sort_values("iteration")["transition_group"]
            .astype(str)
            .str.replace(" -> ", "<br>↓<br>", regex=False)
            .tolist()
        )
        for reason in reason_order:
            reason_df = model_plot_df[model_plot_df["reason"] == reason].sort_values("iteration")
            if reason_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=reason_df["iteration"],
                    y=reason_df["rate"],
                    mode="lines+markers",
                    name=reason,
                    legendgroup=reason,
                    showlegend=col_idx == 1,
                    line=dict(width=3, color=color_map.get(reason)),
                    marker=dict(size=13, symbol=symbol_map.get(reason, "circle")),
                    meta=_model_label(model_name),
                    customdata=reason_df[["count", "total"]].values,
                    hovertemplate="Model: %{meta}<br>Reason: %{fullData.name}<br>Iteration: %{x}<br>Rate: %{y:.1%}<br>Count: %{customdata[0]} / %{customdata[1]}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(
            tickmode="array",
            tickvals=iteration_values,
            ticktext=ticktext,
            range=[min(iteration_values) - 0.2, max(iteration_values) + 0.5],
            row=1,
            col=col_idx,
        )

    y_upper = min(1.0, max(0.05, y_max * 1.15))
    fig.update_yaxes(title_text="Rate", tickformat=".0%", range=[-0.05, y_upper + 0.05])
    fig.update_layout(width=max(1000, 400 * len(plotted_models)), height=520, margin=dict(t=90, b=150, l=85, r=45), legend_title="")
    fig.add_annotation(x=0.5, y=-0.47, xref="paper", yref="paper", text="Web Query Iteration", showarrow=False, font=dict(size=24))
    fig.write_html(output_dir / f"{output_file_name}.html")
    fig = with_paper_style(fig, config=styler(22, 24), legend_pos=(0.8, 1.3))
    fig.write_image(output_dir / f"{output_file_name}.pdf", format="pdf")

    with open(output_dir / f"{output_file_name}_summary.json", "w") as f:
        json.dump(plot_rows, f, indent=2, ensure_ascii=False)
    return plot_df


def print_cross_platform_replay_model_call_outcome_eval_scores(
    evaluator_model="gpt-5.6-luna",
    temperature="0.0",
    base_model_name="gpt-5.3-chat-latest",
    model_names=DEFAULT_MODELS,
):
    eval_dir = Path(f"{OUTPUT_PATH}/metadata/preference_evaluation/{evaluator_model}/{temperature}")
    sample_calls = {}
    for model_name in model_names:
        replay_path = INPUT_DIR / f"{model_name}.json"
        replay_data = _load_replay_json(replay_path)
        for result_key, row in replay_data.items():
            if not isinstance(row, dict) or row.get("skipped_replay"):
                continue
            payload = row.get("auto")
            if not isinstance(payload, dict):
                continue
            response = payload.get("response") or {}
            provider = _infer_provider(model_name, row)
            sample_calls.setdefault(result_key, {})[model_name] = _has_web_tool_call(
                provider,
                response,
            )

    common_result_keys = {
        result_key
        for result_key, calls in sample_calls.items()
        if all(model_name in calls for model_name in model_names)
    }

    eval_by_model = {}
    available_modes_by_model = {}
    for model_name in model_names:
        eval_path = eval_dir / f"{model_name}.csv"
        if not eval_path.exists():
            print(f"Missing evaluation file for `{model_name}`: {eval_path}")
            continue

        model_eval = {}
        available_modes = set()
        with open(eval_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result_key = row.get("result_key")
                if not result_key:
                    continue
                for mode in ["auto", "none", "required", "invivo"]:
                    if (
                        row.get(f"{mode}_factuality_5likert_score") not in (None, "")
                        or row.get(f"{mode}_completeness_5likert_score") not in (None, "")
                        or row.get(f"{mode}_relevance_5likert_score") not in (None, "")
                    ):
                        available_modes.add(mode)
                model_eval[result_key] = row
        eval_by_model[model_name] = model_eval
        available_modes_by_model[model_name] = [
            mode for mode in ["auto", "none", "required", "invivo"] if mode in available_modes
        ]

    for model_name in model_names:
        model_eval = eval_by_model.get(model_name, {})
        available_modes = available_modes_by_model.get(model_name, [])
        buckets = {
            "overall": {
                "count": 0,
                "scores": {},
            },
            "auto_web_call": {
                "count": 0,
                "scores": {},
            },
            "auto_no_web_call": {
                "count": 0,
                "scores": {},
            },
        }
        for bucket in buckets.values():
            for mode in available_modes:
                bucket["scores"][mode] = {
                    "factuality": [],
                    "completeness": [],
                    "relevance": [],
                }

        for result_key in common_result_keys:
            score_row = model_eval.get(result_key, {})
            auto_called_web = bool(
                sample_calls.get(result_key, {}).get(model_name, False)
            )
            buckets["overall"]["count"] += 1
            bucket_key = (
                "auto_web_call" if auto_called_web else "auto_no_web_call"
            )
            buckets[bucket_key]["count"] += 1
            for current_bucket_key in ["overall", bucket_key]:
                for mode in available_modes:
                    for metric_key in ["factuality", "completeness", "relevance"]:
                        score = _safe_float(
                            score_row.get(f"{mode}_{metric_key}_5likert_score")
                        )
                        if score is not None:
                            buckets[current_bucket_key]["scores"][mode][metric_key].append(score)

        if not available_modes:
            continue

        print(_model_label(model_name))
        for bucket_key, bucket_label in [
            ("overall", "overall"),
            ("auto_web_call", "samples with web call in auto"),
            ("auto_no_web_call", "samples with no web call in auto"),
        ]:
            bucket = buckets[bucket_key]
            print(f"  {bucket_label}: n={bucket['count']}")
            for mode in available_modes:
                mode_scores = bucket["scores"][mode]
                print(
                    f"    {mode}: "
                    f"factuality={_mean_with_bootstrap_ci_or_na(mode_scores['factuality'])}, "
                    f"completeness={_mean_with_bootstrap_ci_or_na(mode_scores['completeness'])}, "
                    f"relevance={_mean_with_bootstrap_ci_or_na(mode_scores['relevance'])}"
                )
        print("")


def compute_average_citations_and_retrievals_per_response_for_replays(
    model_names=DEFAULT_MODELS,
    unique=False,
    common_model_names=None,
):
    rows = _extract_rows_for_models(model_names)
    rows = _filter_rows_to_common_samples(rows, model_names)
    common_filter_model_names = (
        list(common_model_names)
        if common_model_names is not None
        else list(model_names)
    )
    sample_calls = {}
    for model_name in model_names:
        replay_path = INPUT_DIR / f"{model_name}.json"
        replay_data = _load_replay_json(replay_path)
        for result_key, row in replay_data.items():
            if not isinstance(row, dict) or row.get("skipped_replay"):
                continue
            payload = row.get("auto")
            if not isinstance(payload, dict):
                continue
            response = payload.get("response") or {}
            provider = _infer_provider(model_name, row)
            sample_calls.setdefault(result_key, {})[model_name] = _has_web_tool_call(
                provider,
                response,
            )

    common_web_result_keys = {
        row.get("result_key")
        for row in _filter_rows_to_samples_with_web_calls_for_all_models(
            _filter_rows_to_common_samples(
                _extract_rows_for_models(common_filter_model_names),
                common_filter_model_names,
            ),
            common_filter_model_names,
        )
        if row.get("result_key") is not None
    }

    def _print_stats(label, model_rows):
        if not model_rows:
            print(label)
            print({"num_responses": 0})
            return

        retrieved_counts = [
            _source_item_count(
                row.get("sources_retrieved", []),
                key="url",
                unique=unique,
            )
            for row in model_rows
        ]
        cited_counts = [
            _source_item_count(
                row.get("sources_cited", []),
                key="url",
                unique=unique,
            )
            for row in model_rows
        ]
        query_trace_counts = [
            sum(
                1
                for query_group in (row.get("web_queries", []) or [])
                for query in (query_group or [])
                if isinstance(query, str) and query.strip()
            )
            for row in model_rows
        ]
        all_web_call_samples = np.ones(len(model_rows), dtype=bool)
        has_web_queries = np.array([count > 0 for count in query_trace_counts], dtype=bool)
        cited_external_counts = []
        cited_internal_counts = []
        retrieved_urls_per_query_trace_counts = []
        for row, query_count in zip(model_rows, query_trace_counts):
            retrieved_sources = _flatten_source_items(row.get("sources_retrieved", []))
            cited_sources = _flatten_source_items(row.get("sources_cited", []))
            retrieved_urls_per_query_trace_counts.append(
                [
                    len(source_group)
                    for source_group in (row.get("sources_retrieved", []) or [])
                    if isinstance(source_group, list)
                ]
            )
            if unique:
                retrieved_urls = {
                    _normalize_url_for_source_matching(src.get("url", ""))
                    for src in retrieved_sources
                    if isinstance(src, dict) and src.get("url")
                }
                cited_urls = [
                    _normalize_url_for_source_matching(src.get("url", ""))
                    for src in cited_sources
                    if isinstance(src, dict) and src.get("url")
                ]
            else:
                retrieved_urls = [
                    _normalize_url_for_source_matching(src.get("url", ""))
                    for src in retrieved_sources
                    if isinstance(src, dict) and src.get("url")
                ]
                cited_urls = [
                    _normalize_url_for_source_matching(src.get("url", ""))
                    for src in cited_sources
                    if isinstance(src, dict) and src.get("url")
                ]
                retrieved_urls = set(retrieved_urls)

            cited_external_count = sum(1 for url in cited_urls if url in retrieved_urls)
            cited_external_counts.append(cited_external_count)
            cited_internal_counts.append(len(cited_urls) - cited_external_count)

        sum_retrievals = float(sum(retrieved_counts))
        sum_citations = float(sum(cited_counts))
        sum_cited_external = int(sum(cited_external_counts))
        sum_cited_internal = int(sum(cited_internal_counts))

        print(label)
        def _build_query_url_counts(sample_retrieved_counts, sample_query_trace_counts):
            query_url_counts = []
            for ret, query_count in zip(sample_retrieved_counts, sample_query_trace_counts):
                if not np.isfinite(query_count) or query_count <= 0:
                    continue
                query_url_counts.extend([ret / query_count] * int(query_count))
            return query_url_counts

        def _print_summary(summary_name, sample_mask, include_query_metrics):
            sample_conv_ids = np.asarray(
                [row.get("conv_id") for row, include_row in zip(model_rows, sample_mask) if include_row],
                dtype=object,
            )
            sample_retrieved_counts = np.asarray(retrieved_counts, dtype=float)[sample_mask]
            sample_cited_counts = np.asarray(cited_counts, dtype=float)[sample_mask]
            sample_query_trace_counts = np.asarray(query_trace_counts, dtype=float)[sample_mask]
            sample_cited_external_counts = np.asarray(cited_external_counts, dtype=float)[sample_mask]
            sample_cited_internal_counts = np.asarray(cited_internal_counts, dtype=float)[sample_mask]
            sum_sample_retrievals = float(np.sum(sample_retrieved_counts))
            sum_sample_citations = float(np.sum(sample_cited_counts))
            sum_sample_cited_external = int(np.sum(sample_cited_external_counts))
            sum_sample_cited_internal = int(np.sum(sample_cited_internal_counts))

            payload = {
                "summary": summary_name,
                "num_responses": int(np.sum(sample_mask)),
                "sum_retrievals": sum_sample_retrievals,
                "avg_retrievals_per_response": (
                    float(np.mean(sample_retrieved_counts))
                    if len(sample_retrieved_counts) > 0
                    else 0.0
                ),
                "sum_citations": sum_sample_citations,
                "avg_citations_per_response": (
                    float(np.mean(sample_cited_counts))
                    if len(sample_cited_counts) > 0
                    else 0.0
                ),
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
                sum_queries_issued = int(np.nansum(sample_query_trace_counts))
                query_url_counts = _build_query_url_counts(
                    sample_retrieved_counts,
                    sample_query_trace_counts,
                )
                payload.update({
                    "sum_queries_issued": sum_queries_issued,
                    "avg_queries_per_response": (
                        float(np.mean(sample_query_trace_counts))
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

            payload["citation_rate_ci_95"] = _bootstrap_ratio_ci_by_conversation(
                sample_conv_ids,
                sample_cited_external_counts,
                sample_retrieved_counts,
            )
            payload["grounding_rate_ci_95"] = _bootstrap_ratio_ci_by_conversation(
                sample_conv_ids,
                sample_cited_external_counts,
                sample_cited_counts,
            )

            print(payload)

        _print_summary(
            "all_samples_with_web_call",
            all_web_call_samples,
            include_query_metrics=False,
        )
        _print_summary(
            "samples_with_web_queries",
            has_web_queries,
            include_query_metrics=True,
        )

    for model_name in model_names:
        model_rows_platform = [
            row
            for row in rows
            if row.get("model") == model_name
            and bool(sample_calls.get(row.get("result_key"), {}).get(model_name, False))
        ]
        print(_model_label(model_name))
        _print_stats("platform web-calling samples", model_rows_platform)
        model_rows_common = [
            row
            for row in model_rows_platform
            if row.get("result_key") in common_web_result_keys
        ]
        _print_stats("common web-calling samples across all common models", model_rows_common)


async def extract_replay_urls_content(
    model_names=DEFAULT_MODELS,
    replay_mode="auto",
    urls_content_path=REPLAY_URLS_CONTENT_PATH,
    force_refresh=False,
    require_all_models_web_call=True,
    common_filter_model_names=None,
):
    from src.response_generation.response_generation import (
        URL_FETCH_CHECKPOINT_EVERY,
        URL_FETCH_TIMEOUT,
        _load_urls_content,
        fetch_url_content,
        logger,
        to_json,
    )

    model_slug = _model_subset_slug(model_names)
    if str(urls_content_path) == str(REPLAY_URLS_CONTENT_PATH):
        urls_content_path = (
            OUTPUT_DIR / f"replay_response_and_sources_url_content__{model_slug}.json"
        )

    filter_model_names = (
        list(common_filter_model_names)
        if common_filter_model_names is not None
        else list(model_names)
    )

    filter_rows = _extract_rows_for_models(filter_model_names, tool_choice=replay_mode)
    filter_rows = _filter_rows_to_common_samples(filter_rows, filter_model_names)
    if require_all_models_web_call:
        filter_rows = _filter_rows_to_samples_with_web_calls_for_all_models(
            filter_rows,
            filter_model_names,
        )
    valid_result_keys = {
        row.get("result_key")
        for row in filter_rows
        if row.get("result_key")
    }

    rows = _extract_rows_for_models(model_names, tool_choice=replay_mode)
    rows = [row for row in rows if row.get("result_key") in valid_result_keys]

    unique_urls = set()
    for row in rows:
        if not bool(row.get("has_web_tool_call", False)):
            continue
        for source in _flatten_source_items(row.get("sources_retrieved", [])):
            if isinstance(source, dict):
                url = _normalize_url_for_source_matching(source.get("url", ""))
                if url:
                    unique_urls.add(url)
        for source in _flatten_source_items(row.get("sources_cited", [])):
            if isinstance(source, dict):
                url = _normalize_url_for_source_matching(source.get("url", ""))
                if url:
                    unique_urls.add(url)

    print(f"Replay URL content extraction: {len(unique_urls)} unique urls")

    url_cache = (
        {}
        if force_refresh
        else _load_urls_content(urls_content_path=urls_content_path, required=False)
    )
    checkpoint_every = max(1, URL_FETCH_CHECKPOINT_EVERY)
    processed_urls = 0

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            for url in tqdm(
                sorted(unique_urls),
                total=len(unique_urls),
                desc=f"Fetching replay URL content ({model_slug})",
            ):
                if force_refresh or url not in url_cache:
                    processed_urls += 1
                    try:
                        url_cache[url] = await asyncio.wait_for(
                            fetch_url_content(
                                url, browser=browser, url_cache=url_cache
                            ),
                            timeout=URL_FETCH_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Replay URL extraction timed out after %.1fs: %s",
                            URL_FETCH_TIMEOUT,
                            url,
                        )
                        url_cache[url] = ""
                        try:
                            await browser.close()
                        except Exception:
                            pass
                        try:
                            browser = await p.chromium.launch(
                                headless=True,
                                args=["--disable-blink-features=AutomationControlled"],
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to relaunch browser after timeout for %s: %s",
                                url,
                                exc,
                            )
                            browser = None
                    if processed_urls % checkpoint_every == 0:
                        logger.info(
                            "Checkpointing replay URL content cache after %s processed URLs to %s",
                            processed_urls,
                            urls_content_path,
                        )
                        to_json(url_cache, str(urls_content_path), indent=2)
        finally:
            if browser is not None:
                await browser.close()

    logger.info(
        "Writing final replay URL content cache with %s entries to %s",
        len(url_cache),
        urls_content_path,
    )
    to_json(url_cache, str(urls_content_path), indent=2)
    return url_cache


def replay_response_source_nli_sentence_based(
    nli_method="bert",
    judge_entailment_min_score=1,
    chunking_method="claim",
    claim_selection_mode="all",
    model_names=DEFAULT_MODELS,
    replay_mode="auto",
    urls_content_path=REPLAY_URLS_CONTENT_PATH,
    force_refresh_urls=False,
    claim_cache_path=REPLAY_CLAIM_CACHE_PATH,
    common_filter_model_names=None,
    save_every=1,
):
    import numpy as np
    import pandas as pd

    from src.response_generation.response_generation import (
        BERT_NLI_MODEL_NAME,
        _claim_cache_key,
        _load_claims_cache,
        _load_urls_content,
        _normalize_claim_selection_mode,
        _normalize_chunking_method,
        _save_claims_cache,
        compute_nli_scores,
        extract_claims_from_text,
    )

    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    if chunking_method != "claim":
        raise ValueError("Replay response-source NLI currently supports chunking_method='claim' only.")
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    model_slug = _model_subset_slug(model_names)

    output_base = OUTPUT_DIR / (
        f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
        f"{claim_selection_mode}__{model_slug}"
    )
    if str(urls_content_path) == str(REPLAY_URLS_CONTENT_PATH):
        urls_content_path = (
            OUTPUT_DIR / f"replay_response_and_sources_url_content__{model_slug}.json"
        )
    if str(claim_cache_path) == str(REPLAY_CLAIM_CACHE_PATH):
        claim_cache_path = (
            OUTPUT_DIR / f"replay_response_source_claim_chunks_cache__{model_slug}.json"
        )

    filter_model_names = (
        list(common_filter_model_names)
        if common_filter_model_names is not None
        else list(model_names)
    )

    filter_rows = _extract_rows_for_models(filter_model_names, tool_choice=replay_mode)
    filter_rows = _filter_rows_to_common_samples(filter_rows, filter_model_names)
    filter_rows = _filter_rows_to_samples_with_web_calls_for_all_models(
        filter_rows, filter_model_names
    )
    valid_result_keys = {
        row.get("result_key")
        for row in filter_rows
        if row.get("result_key")
    }

    rows = _extract_rows_for_models(model_names, tool_choice=replay_mode)
    rows = [
        row
        for row in rows
        if row.get("result_key") in valid_result_keys
        and bool(row.get("has_web_tool_call", False))
    ]

    if not rows:
        print("No replay rows available for response-source NLI.")
        return pd.DataFrame()

    if force_refresh_urls or not Path(urls_content_path).exists():
        asyncio.run(
            extract_replay_urls_content(
                model_names=model_names,
                replay_mode=replay_mode,
                urls_content_path=urls_content_path,
                force_refresh=force_refresh_urls,
                common_filter_model_names=common_filter_model_names,
            )
        )

    urls_content_by_clean_url = {}
    urls_content = _load_urls_content(
        urls_content_path=urls_content_path,
        required=False,
    )
    for url, content in (urls_content or {}).items():
        clean_url = _normalize_url_for_source_matching(url)
        if clean_url:
            urls_content_by_clean_url[clean_url] = content
            urls_content_by_clean_url[clean_url.rstrip("/")] = content

    persisted_claims_cache = _load_claims_cache(cache_path=claim_cache_path)
    claims_cache = {}
    claims_cache_dirty = False
    new_claim_cache_entries = 0

    markdown_citation_pattern = re.compile(r"\[[^\n]{0,120}?\]\((https?://[^)\s]+)\)")

    def _extract_claims(text):
        nonlocal claims_cache_dirty
        nonlocal new_claim_cache_entries

        text = str(text or "").strip()
        if not text:
            return []
        if text in claims_cache:
            return claims_cache[text]

        cache_key = _claim_cache_key(text)
        cached_claims = persisted_claims_cache.get(cache_key, []) or persisted_claims_cache.get(text, [])
        if cached_claims:
            claims_cache[text] = cached_claims
            return cached_claims

        claims = extract_claims_from_text(text)
        if not claims:
            claims = [text]

        claims_cache[text] = claims
        persisted_claims_cache[cache_key] = claims
        claims_cache_dirty = True
        new_claim_cache_entries += 1
        if new_claim_cache_entries % 25 == 0:
            _save_claims_cache(persisted_claims_cache, cache_path=claim_cache_path)
        return claims

    def _claims_with_spans(text, offset=0):
        claims = _extract_claims(text)
        results = []
        cursor = 0
        for claim in claims:
            claim_text = str(claim or "").strip(" -*\t\n")
            if len(claim_text) < 8 or not re.search(r"[A-Za-z]", claim_text):
                continue
            start = text.find(claim_text, cursor)
            if start == -1:
                start = text.find(claim_text)
            end = start + len(claim_text) if start != -1 else -1
            if end != -1:
                cursor = end
            results.append(
                {
                    "text": claim_text,
                    "start": (offset + start) if start != -1 else None,
                    "end": (offset + end) if end != -1 else None,
                    "associated_urls": [],
                    "citation_markers": [],
                }
            )
        return results

    def _append_claim_rows(claim_rows, claims, associated_urls, marker_text):
        cleaned_urls = [
            _normalize_url_for_source_matching(url)
            for url in associated_urls
            if _normalize_url_for_source_matching(url)
        ]
        for claim in claims:
            payload = {
                "sentence": claim["text"],
                "citation_urls": list(dict.fromkeys(cleaned_urls)),
                "citation_markers": [marker_text] if marker_text else [],
                "start": claim.get("start"),
                "end": claim.get("end"),
            }
            claim_rows.append(payload)

    def _extract_markdown_claim_rows(response_text):
        claim_rows = []
        matches = list(markdown_citation_pattern.finditer(response_text or ""))
        previous_end = 0

        for match in matches:
            marker_text = match.group(0)
            marker_urls = [_normalize_url_for_source_matching(match.group(1))]
            chunk_text = response_text[previous_end:match.start()]
            claims = _claims_with_spans(chunk_text, offset=previous_end)

            if claim_selection_mode == "latest_preceding":
                if claims:
                    _append_claim_rows(claim_rows, [claims[-1]], marker_urls, marker_text)
                elif claim_rows:
                    claim_rows[-1]["citation_urls"] = list(
                        dict.fromkeys(claim_rows[-1]["citation_urls"] + marker_urls)
                    )
                    claim_rows[-1]["citation_markers"].append(marker_text)
            else:
                if claims:
                    _append_claim_rows(claim_rows, claims, marker_urls, marker_text)
                elif claim_rows:
                    claim_rows[-1]["citation_urls"] = list(
                        dict.fromkeys(claim_rows[-1]["citation_urls"] + marker_urls)
                    )
                    claim_rows[-1]["citation_markers"].append(marker_text)

            previous_end = match.end()

        tail_claims = _claims_with_spans(response_text[previous_end:], offset=previous_end)
        if tail_claims:
            _append_claim_rows(claim_rows, tail_claims, [], "")
        if not claim_rows:
            _append_claim_rows(claim_rows, _claims_with_spans(response_text), [], "")
        return claim_rows

    def _ranges_overlap(start_a, end_a, start_b, end_b):
        if None in {start_a, end_a, start_b, end_b}:
            return False
        return max(start_a, start_b) < min(end_a, end_b)

    def _extract_claude_claim_rows(response_text, cited_sources, response_text_blocks):
        claims = _claims_with_spans(response_text)
        if not claims:
            return []

        ordered_blocks = [
            block
            for block in (response_text_blocks or [])
            if isinstance(block, dict) and str(block.get("text", "") or "")
        ]
        if ordered_blocks:
            synthetic_parts = []
            for block in ordered_blocks:
                block_text = str(block.get("text", "") or "")
                citation_urls = list(
                    dict.fromkeys(
                        _normalize_url_for_source_matching(citation.get("url", ""))
                        for citation in (block.get("citations", []) or [])
                        if isinstance(citation, dict)
                        and _normalize_url_for_source_matching(citation.get("url", ""))
                    )
                )
                synthetic_parts.append(block_text)
                if citation_urls:
                    synthetic_parts.append(
                        "".join(f" [Claude Citation]({url})" for url in citation_urls)
                    )
            synthetic_response = "".join(synthetic_parts)
            claim_rows = _extract_markdown_claim_rows(synthetic_response)
            if claim_rows:
                return claim_rows

        cited_with_spans = []
        cited_with_block_text = []

        def _normalize_match_text(text):
            return re.sub(r"\s+", " ", str(text or "").strip()).lower()

        for source in cited_sources:
            if not isinstance(source, dict):
                continue
            url = _normalize_url_for_source_matching(source.get("url", ""))
            response_block_text = str(source.get("response_text", "") or "").strip()
            if url and response_block_text:
                cited_with_block_text.append(
                    {
                        "url": url,
                        "response_text": response_block_text,
                        "normalized_response_text": _normalize_match_text(
                            response_block_text
                        ),
                    }
                )
            try:
                start = int(source.get("start_index"))
                end = int(source.get("end_index"))
            except (TypeError, ValueError):
                continue
            if not url or end <= start:
                continue
            cited_with_spans.append(
                {
                    "url": url,
                    "start": start,
                    "end": end,
                }
            )

        if not cited_with_spans and not cited_with_block_text:
            return [
                {
                    "sentence": claim["text"],
                    "citation_urls": [],
                    "citation_markers": [],
                    "start": claim.get("start"),
                    "end": claim.get("end"),
                }
                for claim in claims
            ]

        if not cited_with_spans and cited_with_block_text:
            rows_out = []
            block_payloads = []
            seen_blocks = {}
            for item in cited_with_block_text:
                response_block_text = str(item.get("response_text", "") or "").strip()
                if not response_block_text:
                    continue
                existing = seen_blocks.get(response_block_text)
                if existing is None:
                    existing = {
                        "response_text": response_block_text,
                        "urls": [],
                    }
                    seen_blocks[response_block_text] = existing
                    block_payloads.append(existing)
                if item["url"]:
                    existing["urls"].append(item["url"])

            search_cursor = 0
            for block in block_payloads:
                response_block_text = block["response_text"]
                block_start = response_text.find(response_block_text, search_cursor)
                if block_start == -1:
                    block_start = response_text.find(response_block_text)
                block_offset = block_start if block_start != -1 else 0
                if block_start != -1:
                    search_cursor = block_start + len(response_block_text)

                block_claims = _claims_with_spans(response_block_text, offset=block_offset)
                if not block_claims:
                    block_claims = [
                        {
                            "text": response_block_text,
                            "start": block_offset if block_start != -1 else None,
                            "end": (
                                block_offset + len(response_block_text)
                                if block_start != -1
                                else None
                            ),
                        }
                    ]

                associated_urls = list(dict.fromkeys(block["urls"]))
                for claim in block_claims:
                    rows_out.append(
                        {
                            "sentence": claim["text"],
                            "citation_urls": associated_urls,
                            "citation_markers": [],
                            "start": claim.get("start"),
                            "end": claim.get("end"),
                        }
                    )
            return rows_out

        if claim_selection_mode == "all":
            rows_out = []
            for claim in claims:
                urls = [
                    item["url"]
                    for item in cited_with_spans
                    if _ranges_overlap(
                        claim.get("start"),
                        claim.get("end"),
                        item["start"],
                        item["end"],
                    )
                ]
                rows_out.append(
                    {
                        "sentence": claim["text"],
                        "citation_urls": list(dict.fromkeys(urls)),
                        "citation_markers": [],
                        "start": claim.get("start"),
                        "end": claim.get("end"),
                    }
                )
            return rows_out

        claim_rows = {
            idx: {
                "sentence": claim["text"],
                "citation_urls": [],
                "citation_markers": [],
                "start": claim.get("start"),
                "end": claim.get("end"),
            }
            for idx, claim in enumerate(claims)
        }
        for source in cited_with_spans:
            candidate_idx = None
            candidate_end = None
            for idx, claim in enumerate(claims):
                if _ranges_overlap(
                    claim.get("start"),
                    claim.get("end"),
                    source["start"],
                    source["end"],
                ):
                    claim_end = claim.get("end")
                    if candidate_end is None or (claim_end is not None and claim_end > candidate_end):
                        candidate_idx = idx
                        candidate_end = claim_end
            if candidate_idx is not None:
                claim_rows[candidate_idx]["citation_urls"].append(source["url"])

        for row_payload in claim_rows.values():
            row_payload["citation_urls"] = list(dict.fromkeys(row_payload["citation_urls"]))
        return list(claim_rows.values())

    def _extract_response_claim_rows(row):
        response_text = str(row.get("final_response", "") or "")
        provider = _infer_provider(row.get("model"), row)
        if provider == "claude":
            return _extract_claude_claim_rows(
                response_text,
                row.get("sources_cited", []),
                row.get("response_text_blocks", []),
            )
        return _extract_markdown_claim_rows(response_text)

    def _source_records(row):
        records = []
        seen_keys = set()
        for source_col, source_type in [
            ("sources_cited", "Cited"),
            ("sources_retrieved", "Retrieved"),
        ]:
            for src in _flatten_source_items(row.get(source_col, [])):
                if not isinstance(src, dict):
                    continue
                url = _normalize_url_for_source_matching(src.get("url", ""))
                if not url:
                    continue
                key = (source_type, url)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                records.append(
                    {
                        "url": url,
                        "source_type": source_type,
                        "domain": _normalize_domain_for_top_plots(
                            src.get("domain") or urlparse(url).netloc
                        ),
                        "title": src.get("title", ""),
                    }
                )
        return records

    def _source_content(url):
        clean_url = _normalize_url_for_source_matching(url)
        return str(
            urls_content_by_clean_url.get(
                clean_url,
                urls_content_by_clean_url.get(clean_url.rstrip("/"), ""),
            )
            or ""
        )

    def _load_bert_nli_model():
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(BERT_NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(BERT_NLI_MODEL_NAME)
            model.eval()
            return {
                "torch": torch,
                "tokenizer": tokenizer,
                "model": model,
            }
        except Exception as exc:
            print(f"Could not initialize BERT NLI model {BERT_NLI_MODEL_NAME}: {exc}")
            return None

    bert_nli_model = _load_bert_nli_model() if nli_method == "bert" else None

    def _bert_nli_scores(source_text, sentence):
        if bert_nli_model is None:
            return {"label": "", "confidence": 0.0, "reasoning": ""}
        source_text = str(source_text or "").strip()
        sentence = str(sentence or "").strip()
        if not source_text or not sentence:
            return {"label": "", "confidence": 0.0, "reasoning": ""}
        try:
            torch = bert_nli_model["torch"]
            tokenizer = bert_nli_model["tokenizer"]
            model = bert_nli_model["model"]
            encoded = tokenizer(
                source_text,
                sentence,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            with torch.no_grad():
                logits = model(**encoded).logits[0]
                label_mapping = ["contradiction", "neutral", "entailment"]
                probs = torch.softmax(logits, dim=-1)
                label_id = int(torch.argmax(probs).item())
                confidence = float(probs[label_id].item())
            return {
                "label": label_mapping[label_id],
                "confidence": confidence,
                "reasoning": "",
            }
        except Exception as exc:
            print(f"BERT NLI scoring failed: {exc}")
            return {"label": "", "confidence": 0.0, "reasoning": ""}

    def _nli_label(payload):
        payload = payload if isinstance(payload, dict) else {}
        return str(payload.get("label", "")).strip().lower()

    def _nli_score(payload):
        payload = payload if isinstance(payload, dict) else {}
        try:
            return float(payload.get("confidence", payload.get("score", 0)) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _nli_reasoning(payload):
        payload = payload if isinstance(payload, dict) else {}
        return payload.get("reasoning", payload.get("reason", ""))

    def _score_candidate(source, sentence, source_relation, source_group):
        source_text = _source_content(source["url"])
        if nli_method == "judge":
            nli_judge = (
                compute_nli_scores(source_text, sentence)
                if source_text.strip() and sentence.strip()
                else {"label": "", "confidence": 0.0, "reasoning": ""}
            )
            bert_nli = {"label": "", "confidence": 0.0, "reasoning": ""}
        else:
            nli_judge = {"label": "", "confidence": 0.0, "reasoning": ""}
            bert_nli = _bert_nli_scores(source_text, sentence)

        judge_label = _nli_label(nli_judge)
        judge_score = _nli_score(nli_judge)
        judge_entailed = judge_label == "entailment" and judge_score >= judge_entailment_min_score
        bert_label = _nli_label(bert_nli)
        bert_confidence = _nli_score(bert_nli)
        bert_entailed = bert_label == "entailment"
        attribution_entailed = judge_entailed if nli_method == "judge" else bert_entailed
        source_bucket = {
            "cited_marker": "Associated Citations",
            "other_cited": "Other Citations",
            "retrieved": "Retrieved Sources",
        }.get(source_relation, source_group)
        return {
            "nli_method": nli_method,
            "url": source["url"],
            "domain": source.get("domain", ""),
            "title": source.get("title", ""),
            "source_type": source["source_type"],
            "source_relation": source_relation,
            "source_group": source_group,
            "source_bucket": source_bucket,
            "source_content_chars": len(source_text),
            "judge_nli_label": judge_label,
            "judge_nli_score": judge_score,
            "judge_nli_reasoning": _nli_reasoning(nli_judge),
            "judge_entailed": judge_entailed,
            "bert_nli_label": bert_label,
            "bert_nli_confidence": bert_confidence,
            "bert_nli_reasoning": _nli_reasoning(bert_nli),
            "bert_entailed": bert_entailed,
            "attribution_entailed": attribution_entailed,
            "entailed": attribution_entailed,
        }

    def _candidate_sources(source_records, citation_urls):
        candidates = []
        seen_urls = set()
        citation_urls = {
            _normalize_url_for_source_matching(url)
            for url in (citation_urls or [])
            if _normalize_url_for_source_matching(url)
        }
        cited_sources = [source for source in source_records if source["source_type"] == "Cited"]
        retrieved_sources = [source for source in source_records if source["source_type"] == "Retrieved"]
        cited_by_url = {source["url"]: source for source in cited_sources if source.get("url")}
        retrieved_by_url = {source["url"]: source for source in retrieved_sources if source.get("url")}
        cited_urls = set(cited_by_url)

        def _append_sources(sources, source_relation, source_group):
            for source in sources:
                url = source["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                candidate = source.copy()
                candidate["source_relation"] = source_relation
                candidate["source_group"] = source_group
                candidates.append(candidate)

        marker_sources = [
            cited_by_url[url]
            for url in citation_urls
            if url in cited_by_url
        ]
        marker_sources.extend(
            retrieved_by_url[url]
            for url in citation_urls
            if url in retrieved_by_url and url not in cited_by_url
        )
        _append_sources(marker_sources, "cited_marker", "Cited Sources")

        other_cited_sources = [
            source
            for source in cited_sources
            if source["url"] not in seen_urls
        ]
        _append_sources(other_cited_sources, "other_cited", "Cited Sources")

        residual_retrieved_sources = [
            source
            for source in retrieved_sources
            if source["url"] not in cited_urls and source["url"] not in seen_urls
        ]
        _append_sources(residual_retrieved_sources, "retrieved", "Retrieved Sources")
        return candidates

    def _json_safe(value):
        if isinstance(value, dict):
            return {key: _json_safe(inner_value) for key, inner_value in value.items()}
        if isinstance(value, list):
            return [_json_safe(inner_value) for inner_value in value]
        if isinstance(value, tuple):
            return [_json_safe(inner_value) for inner_value in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    def _checked_source_payload(check):
        return {
            "url": check["url"],
            "domain": check["domain"],
            "title": check["title"],
            "source_type": check["source_type"],
            "source_bucket": check["source_bucket"],
            "source_content_chars": check["source_content_chars"],
            "judge_nli_label": check["judge_nli_label"],
            "judge_nli_score": check["judge_nli_score"],
            "judge_nli_reasoning": check["judge_nli_reasoning"],
            "judge_entailed": check["judge_entailed"],
            "bert_nli_label": check["bert_nli_label"],
            "bert_nli_confidence": check["bert_nli_confidence"],
            "bert_nli_reasoning": check["bert_nli_reasoning"],
            "bert_entailed": check["bert_entailed"],
        }

    def _save_result_checkpoint():
        result_df = pd.DataFrame(result_rows)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(f"{output_base}.csv", index=False)
        result_df.to_pickle(f"{output_base}.pkl")
        with open(f"{output_base}.json", "w") as f:
            json.dump(
                [
                    {key: _json_safe(value) for key, value in record.items()}
                    for record in result_df.to_dict(orient="records")
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
        return result_df

    result_rows = []
    for sample_index, row in enumerate(tqdm(rows, total=len(rows))):
        response_text = str(row.get("final_response", "") or "")
        claim_rows = _extract_response_claim_rows(row)
        sources = _source_records(row)
        for sentence_index, claim_payload in enumerate(claim_rows):
            sentence = claim_payload["sentence"]
            citation_urls = claim_payload["citation_urls"]
            candidates = _candidate_sources(sources, citation_urls)
            checked_sources = []
            entailed_check = None

            marker_candidates = [
                candidate
                for candidate in candidates
                if candidate["source_relation"] == "cited_marker"
            ]
            other_cited_candidates = [
                candidate
                for candidate in candidates
                if candidate["source_relation"] == "other_cited"
            ]
            retrieved_candidates = [
                candidate
                for candidate in candidates
                if candidate["source_relation"] == "retrieved"
            ]

            def _evaluate_candidates(candidate_group, stop_on_entailment=False):
                checks = []
                for candidate in candidate_group:
                    check = _score_candidate(
                        candidate,
                        sentence,
                        candidate["source_relation"],
                        candidate["source_group"],
                    )
                    checks.append(check)
                    if stop_on_entailment and check["attribution_entailed"]:
                        break
                return checks

            def _first_entailed(checks):
                for check in checks:
                    if check["attribution_entailed"]:
                        return check
                return None

            marker_checks = _evaluate_candidates(marker_candidates)
            checked_sources.extend(marker_checks)
            entailed_check = _first_entailed(marker_checks)

            if entailed_check is None:
                other_checks = _evaluate_candidates(other_cited_candidates, stop_on_entailment=True)
                checked_sources.extend(other_checks)
                entailed_check = _first_entailed(other_checks)

            if entailed_check is None:
                retrieved_checks = _evaluate_candidates(retrieved_candidates, stop_on_entailment=True)
                checked_sources.extend(retrieved_checks)
                entailed_check = _first_entailed(retrieved_checks)

            if entailed_check is None:
                entailed_check = {
                    "url": "",
                    "domain": "",
                    "title": "",
                    "source_type": "Unknown",
                    "source_relation": "Unknown",
                    "source_group": "Unknown",
                    "source_bucket": "Parametric Knowledge",
                    "source_content_chars": 0,
                    "judge_nli_label": "",
                    "judge_nli_score": 0,
                    "judge_nli_reasoning": "",
                    "judge_entailed": False,
                    "bert_nli_label": "",
                    "bert_nli_confidence": 0.0,
                    "bert_nli_reasoning": "",
                    "bert_entailed": False,
                    "attribution_entailed": False,
                    "entailed": False,
                }

            result_rows.append(
                {
                    "sample": sample_index,
                    "model": row.get("model"),
                    "provider": row.get("provider"),
                    "tool_choice": row.get("tool_choice"),
                    "result_key": row.get("result_key"),
                    "sample_idx": row.get("sample_idx"),
                    "conv_id": row.get("conv_id"),
                    "turn_id": row.get("turn_id"),
                    "user_prompt": row.get("user_prompt"),
                    "response_text": response_text,
                    "response_chunk_index": sentence_index,
                    "response_chunk_text": sentence,
                    "citation_urls": citation_urls,
                    "citation_markers": claim_payload["citation_markers"],
                    "checked_source_count": len(checked_sources),
                    "entailment_source_bucket": entailed_check["source_bucket"],
                    "entailment_source_type": entailed_check["source_type"],
                    "entailed_url": entailed_check["url"],
                    "entailed_domain": entailed_check["domain"],
                    "entailed_title": entailed_check["title"],
                    "judge_nli_label": entailed_check["judge_nli_label"],
                    "judge_nli_score": entailed_check["judge_nli_score"],
                    "judge_nli_reasoning": entailed_check["judge_nli_reasoning"],
                    "judge_entailed": entailed_check["judge_entailed"],
                    "bert_nli_label": entailed_check["bert_nli_label"],
                    "bert_nli_confidence": entailed_check["bert_nli_confidence"],
                    "bert_nli_reasoning": entailed_check["bert_nli_reasoning"],
                    "bert_entailed": entailed_check["bert_entailed"],
                    "attribution_entailed": entailed_check["attribution_entailed"],
                    "entailed": entailed_check["entailed"],
                    "Unknown": entailed_check["source_group"] == "Unknown",
                    "checked_sources": [_checked_source_payload(check) for check in checked_sources],
                }
            )

        if save_every and (sample_index + 1) % save_every == 0:
            _save_result_checkpoint()

    if claims_cache_dirty:
        _save_claims_cache(persisted_claims_cache, cache_path=claim_cache_path)

    result_df = _save_result_checkpoint()
    return result_df


def response_source_nli_sentence_based_for_replays(
    nli_method="bert",
    judge_entailment_min_score=1,
    chunking_method="claim",
    claim_selection_mode="all",
    model_names=DEFAULT_MODELS,
    replay_mode="auto",
    urls_content_path=REPLAY_URLS_CONTENT_PATH,
    force_refresh_urls=False,
    claim_cache_path=REPLAY_CLAIM_CACHE_PATH,
    common_filter_model_names=None,
    save_every=1,
):
    return replay_response_source_nli_sentence_based(
        nli_method=nli_method,
        judge_entailment_min_score=judge_entailment_min_score,
        chunking_method=chunking_method,
        claim_selection_mode=claim_selection_mode,
        model_names=model_names,
        replay_mode=replay_mode,
        urls_content_path=urls_content_path,
        force_refresh_urls=force_refresh_urls,
        claim_cache_path=claim_cache_path,
        common_filter_model_names=common_filter_model_names,
        save_every=save_every,
    )


def plot_response_source_nli_sentence_based_for_replays(
    output_base=None,
    file_name=None,
    nli_method="bert",
    chunking_method="claim",
    claim_selection_mode="all",
    model_names=DEFAULT_MODELS,
    replay_mode="auto",
    common_filter_model_names=None,
    output_dir=PLOT_OUTPUT_DIR / "response_generation",
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from src.utils.figure_style import with_paper_style, styler
    from src.response_generation.response_generation import (
        _normalize_claim_selection_mode,
        _normalize_chunking_method,
    )

    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    if chunking_method != "claim":
        raise ValueError("Replay sentence-based NLI plots currently support chunking_method='claim' only.")
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)

    modes_to_plot = ["all", "latest_preceding"]
    model_slug = _model_subset_slug(model_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_order = [
        "Associated Citations",
        "Other Citations",
        "Search Results",
        "Parametric Knowledge",
    ]
    color_map = {
        "Associated Citations": "#EF553B",
        "Other Citations": "#AB63FA",
        "Search Results": "#636EFA",
        "Parametric Knowledge": "#7F7F7F",
    }
    mode_label_map = {
        "all": "All Claims",
        "latest_preceding": "Latest Claim Before Citation",
    }
    requested_models = list(model_names)

    def _replay_nli_output_base(mode):
        if output_base is not None:
            return Path(output_base)
        return OUTPUT_DIR / (
            f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
            f"{mode}__{model_slug}"
        )

    def _citation_urls_key(value):
        if isinstance(value, (list, tuple)):
            parsed = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return tuple()
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    break
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    parsed = None
            if parsed is None:
                return tuple()
        else:
            return tuple()

        if not isinstance(parsed, (list, tuple)):
            return tuple()
        return tuple(str(item) for item in parsed if str(item or "").strip())

    def _prepare_claim_mode_df(df, mode):
        df = df.copy()
        if "chunking_method" in df.columns:
            df = df[df["chunking_method"].fillna("claim") == "claim"].copy()
        if len(df) == 0 or mode == "all":
            return df

        sort_cols = [
            col
            for col in ["sample", "response_chunk_index"]
            if col in df.columns
        ]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable").copy()

        if "citation_urls" not in df.columns:
            return df.iloc[0:0].copy()

        refs_key_series = df["citation_urls"].apply(_citation_urls_key)
        nonempty_refs_mask = refs_key_series.apply(bool)
        if not bool(nonempty_refs_mask.any()):
            return df.iloc[0:0].copy()

        sample_series = (
            df["sample"]
            if "sample" in df.columns
            else pd.Series([0] * len(df), index=df.index)
        )
        model_series = (
            df["model"].astype(str)
            if "model" in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )
        prev_sample = sample_series.shift(1)
        prev_model = model_series.shift(1)
        prev_refs = refs_key_series.shift(1)
        same_as_prev = (
            nonempty_refs_mask
            & (sample_series == prev_sample)
            & (model_series == prev_model)
            & (refs_key_series == prev_refs)
        )
        run_id = (~same_as_prev).cumsum()
        latest_df = df.loc[nonempty_refs_mask].copy()
        latest_df["_run_id"] = run_id[nonempty_refs_mask].values
        latest_df = (
            latest_df.groupby("_run_id", sort=False)
            .tail(1)
            .drop(columns=["_run_id"])
            .copy()
        )
        if sort_cols:
            latest_df = latest_df.sort_values(sort_cols, kind="stable").copy()
        return latest_df

    def _load_replay_nli_df(mode):
        resolved_base = _replay_nli_output_base(mode)
        csv_path = Path(f"{resolved_base}.csv")
        pkl_path = Path(f"{resolved_base}.pkl")
        json_path = Path(f"{resolved_base}.json")
        if csv_path.exists():
            return _prepare_claim_mode_df(pd.read_csv(csv_path), mode)
        if pkl_path.exists():
            return _prepare_claim_mode_df(pd.read_pickle(pkl_path), mode)
        if json_path.exists():
            with open(json_path, "r") as f:
                return _prepare_claim_mode_df(pd.DataFrame(json.load(f)), mode)

        per_model_frames = []
        missing_models = []
        for model_name in requested_models:
            model_base = OUTPUT_DIR / (
                f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
                f"all__{str(model_name).replace('.', '-')}"
            )
            model_csv_path = Path(f"{model_base}.csv")
            model_pkl_path = Path(f"{model_base}.pkl")
            model_json_path = Path(f"{model_base}.json")
            if model_csv_path.exists():
                model_df = pd.read_csv(model_csv_path)
            elif model_pkl_path.exists():
                model_df = pd.read_pickle(model_pkl_path)
            elif model_json_path.exists():
                with open(model_json_path, "r") as f:
                    model_df = pd.DataFrame(json.load(f))
            else:
                missing_models.append(model_name)
                continue
            per_model_frames.append(model_df)

        if per_model_frames:
            combined_df = pd.concat(per_model_frames, ignore_index=True)
            return _prepare_claim_mode_df(combined_df, mode)

        raise FileNotFoundError(
            f"Could not find replay NLI output for {resolved_base}. "
            f"Missing per-model files for: {missing_models}"
        )

    summary_frames = []
    for mode in modes_to_plot:
        try:
            sentence_df = _load_replay_nli_df(mode)
        except FileNotFoundError as exc:
            print(f"Skipping replay sentence-based NLI plot for claim_selection_mode={mode}: {exc}")
            continue

        if len(sentence_df) == 0:
            continue

        if "chunking_method" in sentence_df.columns:
            sentence_df = sentence_df[
                sentence_df["chunking_method"].fillna("claim") == chunking_method
            ].copy()
        if "tool_choice" in sentence_df.columns:
            sentence_df = sentence_df[
                sentence_df["tool_choice"].fillna("").astype(str) == str(replay_mode)
            ].copy()
        if len(sentence_df) == 0:
            continue

        if common_filter_model_names is not None and "result_key" in sentence_df.columns:
            filter_rows = _extract_rows_for_models(
                common_filter_model_names,
                tool_choice=replay_mode,
            )
            filter_rows = _filter_rows_to_common_samples(
                filter_rows, common_filter_model_names
            )
            filter_rows = _filter_rows_to_samples_with_web_calls_for_all_models(
                filter_rows, common_filter_model_names
            )
            valid_result_keys = {
                row.get("result_key")
                for row in filter_rows
                if row.get("result_key")
            }
            sentence_df = sentence_df[
                sentence_df["result_key"].astype(str).isin(valid_result_keys)
            ].copy()
            if len(sentence_df) == 0:
                continue

        mode_summary_frames = []
        for model_name in model_names:
            model_df = sentence_df[sentence_df["model"].astype(str) == str(model_name)].copy()
            if len(model_df) == 0:
                continue

            source_buckets = (
                model_df["entailment_source_bucket"]
                .fillna("Parametric Knowledge")
                .replace(
                    {
                        "": "Parametric Knowledge",
                        "Unexplained": "Parametric Knowledge",
                        "Unknown": "Parametric Knowledge",
                        "unknown": "Parametric Knowledge",
                        "Marked Citations": "Associated Citations",
                        "Retrieved Sources": "Search Results"
                    }
                )
            )
            sentence_weights = pd.Series([1] * len(model_df), index=model_df.index)
            total_sentences = float(sentence_weights.sum())
            if total_sentences <= 0:
                continue

            counts = (
                pd.DataFrame(
                    {
                        "entailment_source_bucket": source_buckets,
                        "sentence_weight": sentence_weights,
                    }
                )
                .groupby("entailment_source_bucket")["sentence_weight"]
                .sum()
            )
            model_summary_df = pd.DataFrame(
                {
                    "entailment_source_bucket": source_order,
                    "sentence_count": [
                        int(counts.get(source_bucket, 0))
                        for source_bucket in source_order
                    ],
                }
            )
            model_summary_df["sentence_rate"] = (
                model_summary_df["sentence_count"] / total_sentences
            )
            model_summary_df["total_sentence_count"] = int(total_sentences)
            model_summary_df["claim_selection_mode"] = mode
            model_summary_df["model"] = model_name
            model_summary_df["model_label"] = _model_label(model_name)
            mode_summary_frames.append(model_summary_df)

        if not mode_summary_frames:
            continue

        mode_summary_df = pd.concat(mode_summary_frames, ignore_index=True)
        mode_file_name = (
            f"{file_name}_{mode}"
            if file_name is not None
            else f"replay_response_source_nli_sentence_based_{nli_method}_summary_{chunking_method}_{mode}__{model_slug}"
        )
        mode_summary_df.to_csv(output_dir / f"{mode_file_name}.csv", index=False)
        with open(output_dir / f"{mode_file_name}.json", "w") as f:
            json.dump(
                mode_summary_df.to_dict(orient="records"),
                f,
                indent=2,
                ensure_ascii=False,
            )

        model_order = [
            model_name
            for model_name in model_names
            if model_name in set(mode_summary_df["model"].astype(str))
        ]
        model_labels = [_model_label(model_name) for model_name in model_order]
        total_by_model = {}
        for model_name in model_order:
            model_rows = mode_summary_df[mode_summary_df["model"] == model_name]
            total_by_model[model_name] = int(model_rows["total_sentence_count"].iloc[0])

        fig = go.Figure()
        for source_bucket in source_order:
            rates = []
            counts = []
            totals = []
            for model_name in model_order:
                bucket_rows = mode_summary_df[
                    (mode_summary_df["model"] == model_name)
                    & (mode_summary_df["entailment_source_bucket"] == source_bucket)
                ]
                if len(bucket_rows) == 0:
                    rates.append(0.0)
                    counts.append(0)
                    totals.append(total_by_model.get(model_name, 0))
                else:
                    rates.append(float(bucket_rows["sentence_rate"].iloc[0]))
                    counts.append(int(bucket_rows["sentence_count"].iloc[0]))
                    totals.append(int(bucket_rows["total_sentence_count"].iloc[0]))
            fig.add_trace(
                go.Bar(
                    x=model_labels,
                    y=rates,
                    name=source_bucket,
                    marker_color=color_map[source_bucket],
                    text=[f"{rate:.1%}" if rate > 0 else "" for rate in rates],
                    textposition="inside",
                    textfont=dict(color="white"),
                    customdata=np.column_stack([counts, totals]),
                    hovertemplate=(
                        "Model: %{x}<br>"
                        "Source bucket: %{fullData.name}<br>"
                        "Claim rate: %{y:.1%}<br>"
                        "Count: %{customdata[0]} / %{customdata[1]}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            barmode="stack",
            xaxis_title="Replay Model",
            yaxis_title="Rate of Response Claims",
            legend_title="Source",
            title=mode_label_map.get(mode, mode),
        )
        fig.update_xaxes(categoryorder="array", categoryarray=model_labels)
        fig.update_yaxes(range=[0, 1], tickformat=".0%")

        fig.write_html(output_dir / f"{mode_file_name}.html")
        paper_fig = with_paper_style(
            fig,
            config=styler(24, 22),
            legend_pos=(0.8, 1.5),
        )
        paper_fig.write_image(output_dir / f"{mode_file_name}.pdf", format="pdf")

        summary_frames.append(mode_summary_df.assign(plot_file_name=mode_file_name))

    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)


def plot_response_source_nli_sentence_based_judge_for_replays(
    output_base=None,
    file_name=None,
    chunking_method="claim",
    claim_selection_mode="all",
    model_names=DEFAULT_MODELS,
    replay_mode="auto",
    common_filter_model_names=None,
    output_dir=PLOT_OUTPUT_DIR / "response_generation",
):
    from src.response_generation.response_generation import (
        _normalize_claim_selection_mode,
        _normalize_chunking_method,
    )

    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    if file_name is None:
        file_name = "replay_response_source_nli_sentence_based_judge_summary"
        if chunking_method != "citation_marker":
            file_name = f"{file_name}_{chunking_method}"
    return plot_response_source_nli_sentence_based_for_replays(
        output_base=output_base,
        file_name=file_name,
        nli_method="judge",
        chunking_method=chunking_method,
        claim_selection_mode=claim_selection_mode,
        model_names=model_names,
        replay_mode=replay_mode,
        common_filter_model_names=common_filter_model_names,
        output_dir=output_dir,
    )


def response_source_nli_sentence_based_factuality_for_replays(
    input_path=None,
    output_base=None,
    factuality_model_name=None,
    output_suffix="_factuality",
    checkpoint_every=10,
    model_names=DEFAULT_MODELS,
    nli_method="judge",
    chunking_method="claim",
    claim_selection_mode="all",
):
    import pandas as pd

    from src.response_generation.response_generation import FACTUALITY_JUDGE_MODEL, _json_safe, evaluate_claim_factuality

    factuality_model_name = factuality_model_name or FACTUALITY_JUDGE_MODEL
    model_slug = _model_subset_slug(model_names)

    def _default_input_path():
        return OUTPUT_DIR / (
            f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
            f"{claim_selection_mode}__{model_slug}.json"
        )

    def _load_records_from_path(path):
        path = Path(path)
        with open(path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list at {path}")
        return payload, str(path)

    def _load_records_from_per_model_files():
        frames = []
        resolved_paths = []
        for replay_model in model_names:
            model_path = OUTPUT_DIR / (
                f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
                f"{claim_selection_mode}__{str(replay_model).replace('.', '-')}.json"
            )
            if not model_path.exists():
                continue
            with open(model_path, "r") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON list at {model_path}")
            frames.extend(payload)
            resolved_paths.append(str(model_path))
        if not frames:
            raise FileNotFoundError(
                "Could not find replay response-source NLI records for the requested models."
            )
        return frames, ",".join(resolved_paths)

    if input_path is not None:
        records, resolved_input_path = _load_records_from_path(input_path)
    else:
        default_path = _default_input_path()
        if default_path.exists():
            records, resolved_input_path = _load_records_from_path(default_path)
        else:
            records, resolved_input_path = _load_records_from_per_model_files()

    if input_path:
        base_without_ext = os.path.splitext(resolved_input_path)[0]
    else:
        base_without_ext = (
            str(output_base)
            if output_base is not None
            else str(os.path.splitext(str(_default_input_path()))[0])
        )

    factuality_output_base = f"{base_without_ext}{output_suffix}"
    factuality_output_dir = os.path.dirname(factuality_output_base)
    if factuality_output_dir:
        os.makedirs(factuality_output_dir, exist_ok=True)

    claim_cache = {}
    enriched_records = []
    checkpoint_every = max(1, int(checkpoint_every or 1))

    def _save_factuality_checkpoint(records_to_save):
        factuality_df = pd.DataFrame(records_to_save)
        factuality_df.to_csv(f"{factuality_output_base}.csv", index=False)
        factuality_df.to_pickle(f"{factuality_output_base}.pkl")
        json_records = [
            {key: _json_safe(value) for key, value in record.items()}
            for record in factuality_df.to_dict(orient="records")
        ]
        with open(f"{factuality_output_base}.json", "w") as f:
            json.dump(json_records, f, indent=2, ensure_ascii=False)
        return factuality_df

    for record_index, record in enumerate(tqdm(records, total=len(records)), start=1):
        if not isinstance(record, dict):
            continue
        claim = str(record.get("response_chunk_text", "") or "").strip()
        if not claim:
            continue
        user_query = str(record.get("user_prompt", "") or "").strip()
        cache_key = (user_query, claim)
        if cache_key in claim_cache:
            factuality = claim_cache[cache_key]
        else:
            factuality = evaluate_claim_factuality(
                claim,
                user_query=user_query,
                model_name=factuality_model_name,
            )
            claim_cache[cache_key] = factuality

        enriched_records.append(
            {
                "model": str(record.get("model", "") or "").strip(),
                "provider": str(record.get("provider", "") or "").strip(),
                "result_key": str(record.get("result_key", "") or "").strip(),
                "sample_idx": record.get("sample_idx"),
                "conv_id": str(record.get("conv_id", "") or "").strip(),
                "turn_id": str(record.get("turn_id", "") or "").strip(),
                "user_prompt": user_query,
                "claim": claim,
                "factuality_label": factuality.get("label", ""),
                "factuality_score": factuality.get("score", 0.0),
                "factuality_reasoning": factuality.get("reasoning", ""),
                "factuality_raw_response": factuality.get("raw_response", ""),
                "factuality_model": factuality.get("model", factuality_model_name),
            }
        )
        if record_index % checkpoint_every == 0:
            _save_factuality_checkpoint(enriched_records)

    factuality_df = _save_factuality_checkpoint(enriched_records)
    return factuality_df


def summarize_response_source_nli_sentence_based_factuality_for_replays(
    input_path=None,
    factuality_input_path=None,
    model_names=DEFAULT_MODELS,
    nli_method="judge",
    chunking_method="claim",
    claim_selection_mode="all",
    n_boot=1000,
    random_state=42,
):
    import numpy as np
    import pandas as pd

    model_slug = _model_subset_slug(model_names)

    if input_path is None:
        input_path = OUTPUT_DIR / (
            f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
            f"{claim_selection_mode}__{model_slug}.json"
        )
    input_path = Path(input_path)

    if factuality_input_path is None:
        base_without_ext, _ = os.path.splitext(str(input_path))
        factuality_input_path = f"{base_without_ext}_factuality.json"

    def _load_list_json(path):
        with open(path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list at {path}")
        return payload

    if input_path.exists():
        source_records = _load_list_json(input_path)
    else:
        source_records = []
        for replay_model in model_names:
            model_path = OUTPUT_DIR / (
                f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
                f"{claim_selection_mode}__{str(replay_model).replace('.', '-')}.json"
            )
            if model_path.exists():
                source_records.extend(_load_list_json(model_path))
        if not source_records:
            raise FileNotFoundError(f"No replay NLI source records found for {model_names}")

    factuality_records = _load_list_json(factuality_input_path)

    factuality_lookup = {}
    for record in factuality_records:
        if not isinstance(record, dict):
            continue
        key = (
            str(record.get("model", "") or "").strip(),
            str(record.get("conv_id", "") or "").strip(),
            str(record.get("turn_id", "") or "").strip(),
            str(record.get("claim", "") or "").strip(),
        )
        if not key[3]:
            continue
        factuality_lookup[key] = record

    replay_rows = _extract_rows_for_models(model_names, tool_choice="auto")
    retrieved_url_lookup = {}
    for row in replay_rows:
        result_key = str(row.get("result_key", "") or "").strip()
        if not result_key:
            continue
        retrieved_urls = {
            _normalize_url_for_source_matching(src.get("url", ""))
            for src in _flatten_source_items(row.get("sources_retrieved", []))
            if isinstance(src, dict) and src.get("url")
        }
        retrieved_url_lookup[(str(row.get("model", "") or "").strip(), result_key)] = (
            retrieved_urls
        )

    source_group_order = [
        "Associated Citations",
        "Other Citations",
        "Retrieved-not-cited",
        "Parametric Knowledge",
    ]
    citation_retrieval_order = ["Cited Retrieved", "Cited Parametric"]

    def _normalize_source_group(raw_bucket):
        raw_bucket = str(raw_bucket or "").strip()
        if raw_bucket == "Marked Citations":
            return "Associated Citations"
        if raw_bucket == "Other Citations":
            return "Other Citations"
        if raw_bucket == "Retrieved Sources":
            return "Retrieved-not-cited"
        if raw_bucket in {"", "Unexplained", "Unknown", "unknown"}:
            return "Parametric Knowledge"
        return raw_bucket

    def _bootstrap_group_mean_cis(group_df, group_col, group_order):
        if len(group_df) == 0:
            return {}

        work_df = group_df.copy()
        response_key_cols = ["model", "conv_id", "turn_id"]
        for col in response_key_cols:
            if col not in work_df.columns:
                work_df[col] = ""
            work_df[col] = work_df[col].astype(str)

        response_keys = list(
            work_df[response_key_cols]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if not response_keys:
            return {}

        response_groups = {
            key: value.copy()
            for key, value in work_df.groupby(
                response_key_cols, dropna=False, sort=False
            )
        }

        rng = np.random.default_rng(random_state)
        boot_means = {group_name: [] for group_name in group_order}

        for _ in range(max(1, int(n_boot or 1))):
            sampled_indices = rng.integers(0, len(response_keys), size=len(response_keys))
            sampled_frames = [response_groups[response_keys[idx]] for idx in sampled_indices]
            boot_df = pd.concat(sampled_frames, ignore_index=True)
            for group_name in group_order:
                subset = pd.to_numeric(
                    boot_df.loc[boot_df[group_col] == group_name, "factuality_score"],
                    errors="coerce",
                ).dropna()
                if len(subset) == 0:
                    boot_means[group_name].append(np.nan)
                else:
                    boot_means[group_name].append(float(subset.mean()))

        ci_lookup = {}
        for group_name, values in boot_means.items():
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                ci_lookup[group_name] = (np.nan, np.nan)
            else:
                ci_lookup[group_name] = (
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                )
        return ci_lookup

    merged_rows = []
    for record in source_records:
        if not isinstance(record, dict):
            continue
        claim = str(record.get("response_chunk_text", "") or "").strip()
        if not claim:
            continue
        model_name = str(record.get("model", "") or "").strip()
        join_key = (
            model_name,
            str(record.get("conv_id", "") or "").strip(),
            str(record.get("turn_id", "") or "").strip(),
            claim,
        )
        factuality_record = factuality_lookup.get(join_key, {})
        source_group = _normalize_source_group(record.get("entailment_source_bucket", ""))
        if source_group not in source_group_order:
            continue

        entailed_url = _normalize_url_for_source_matching(
            str(record.get("entailed_url", "") or "").strip()
        )
        retrieved_urls = retrieved_url_lookup.get(
            (model_name, str(record.get("result_key", "") or "").strip()),
            set(),
        )
        cited_retrieval_bucket = ""
        if source_group in {"Associated Citations", "Other Citations"}:
            cited_retrieval_bucket = (
                "Cited Retrieved" if entailed_url and entailed_url in retrieved_urls else "Cited Parametric"
            )
        score = pd.to_numeric(
            factuality_record.get("factuality_score", np.nan),
            errors="coerce",
        )
        merged_rows.append(
            {
                "model": model_name,
                "model_display": _model_label(model_name),
                "conv_id": join_key[1],
                "turn_id": join_key[2],
                "claim": claim,
                "source_group": source_group,
                "citation_retrieval_bucket": cited_retrieval_bucket,
                "entailed_url": entailed_url,
                "factuality_score": score,
            }
        )

    summary_df = pd.DataFrame(merged_rows)
    if summary_df.empty:
        raise ValueError("No joined replay records found between source and factuality files.")

    def _summarize(group_df, model_display, group_col, group_order, level_label):
        if len(group_df) == 0:
            return pd.DataFrame()
        ci_lookup = _bootstrap_group_mean_cis(group_df, group_col, group_order)
        total_chunks = float(len(group_df))
        grouped = (
            group_df.groupby(group_col, dropna=False)
            .agg(
                chunk_count=("claim", "size"),
                avg_factuality_score=("factuality_score", "mean"),
            )
            .reindex(group_order, fill_value=0)
            .reset_index()
        )
        grouped["chunk_share"] = (
            grouped["chunk_count"].astype(float) / total_chunks
            if total_chunks > 0
            else 0.0
        )
        grouped["ci_low"] = grouped[group_col].map(
            lambda key: ci_lookup.get(key, (np.nan, np.nan))[0]
        )
        grouped["ci_high"] = grouped[group_col].map(
            lambda key: ci_lookup.get(key, (np.nan, np.nan))[1]
        )
        grouped["avg_factuality_score"] = pd.to_numeric(
            grouped["avg_factuality_score"],
            errors="coerce",
        ).round(4)
        grouped["chunk_share"] = pd.to_numeric(
            grouped["chunk_share"],
            errors="coerce",
        ).round(4)
        grouped["ci_low"] = pd.to_numeric(grouped["ci_low"], errors="coerce").round(4)
        grouped["ci_high"] = pd.to_numeric(grouped["ci_high"], errors="coerce").round(4)
        grouped["model_display"] = model_display
        grouped["n_boot"] = int(max(1, int(n_boot or 1)))
        grouped["n_responses"] = int(
            len(group_df[["model", "conv_id", "turn_id"]].drop_duplicates())
        )
        grouped["summary_level"] = level_label
        return grouped

    summary_frames = []
    summary_frames.append(
        _summarize(
            summary_df,
            "All",
            "source_group",
            source_group_order,
            "source_group",
        )
    )
    all_cited_df = summary_df[
        summary_df["citation_retrieval_bucket"].isin(citation_retrieval_order)
    ].copy()
    if len(all_cited_df) > 0:
        summary_frames.append(
            _summarize(
                all_cited_df,
                "All",
                "citation_retrieval_bucket",
                citation_retrieval_order,
                "citation_retrieval_bucket",
            )
        )

    model_display_order = [_model_label(model_name) for model_name in model_names]
    for model_name in model_names:
        display_name = _model_label(model_name)
        model_df = summary_df[summary_df["model"] == model_name].copy()
        if len(model_df) == 0:
            continue
        summary_frames.append(
            _summarize(
                model_df,
                display_name,
                "source_group",
                source_group_order,
                "source_group",
            )
        )
        model_cited_df = model_df[
            model_df["citation_retrieval_bucket"].isin(citation_retrieval_order)
        ].copy()
        if len(model_cited_df) > 0:
            summary_frames.append(
                _summarize(
                    model_cited_df,
                    display_name,
                    "citation_retrieval_bucket",
                    citation_retrieval_order,
                    "citation_retrieval_bucket",
                )
            )

    combined_summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    combined_summary["model_display"] = pd.Categorical(
        combined_summary["model_display"],
        categories=["All"] + model_display_order,
        ordered=True,
    )
    combined_summary["source_group"] = pd.Categorical(
        combined_summary.get("source_group"),
        categories=source_group_order,
        ordered=True,
    )
    combined_summary["citation_retrieval_bucket"] = pd.Categorical(
        combined_summary.get("citation_retrieval_bucket"),
        categories=citation_retrieval_order,
        ordered=True,
    )
    combined_summary = combined_summary.sort_values(
        ["model_display", "summary_level", "source_group", "citation_retrieval_bucket"],
        kind="stable",
    ).reset_index(drop=True)

    print(combined_summary.to_string(index=False))
    return combined_summary


def evaluate_claude_associated_citation_bucket_alignment_for_replays(
    input_path=None,
    nli_method="judge",
    chunking_method="claim",
    claim_selection_mode="all",
    model_name="claude-sonnet-4-6",
    replay_path=None,
    replay_mode="auto",
    output_dir=PLOT_OUTPUT_DIR / "response_generation",
):
    import pandas as pd
    import plotly.graph_objects as go

    from src.utils.figure_style import with_paper_style, styler
    from src.response_generation.response_generation import (
        _normalize_claim_selection_mode,
        _normalize_chunking_method,
    )

    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    if chunking_method != "claim":
        raise ValueError("Claude citation alignment currently supports chunking_method='claim' only.")
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path is None:
        model_slug = str(model_name).replace(".", "-")
        input_path = OUTPUT_DIR / (
            f"replay_response_source_nli_sentence_based_{nli_method}_{chunking_method}_"
            f"{claim_selection_mode}__{model_slug}.json"
        )
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Replay NLI file not found: {input_path}")

    with open(input_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(
            f"Replay NLI file `{input_path}` did not parse to a list; got "
            f"{type(payload).__name__}."
        )
    df = pd.DataFrame(payload)
    if len(df) == 0:
        print("No replay NLI rows available for Claude citation-alignment evaluation.")
        return {}

    df = df[df["model"].astype(str) == str(model_name)].copy()
    if len(df) == 0:
        print(f"No rows found for model {model_name} in {input_path}.")
        return {}

    if replay_path is None:
        replay_path = INPUT_DIR / f"{model_name}.json"
    replay_payload = _load_replay_json(replay_path)

    def _normalize_url_list(value):
        if isinstance(value, list):
            urls = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, list):
                        urls = parsed
                        break
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
            else:
                return []
        else:
            return []

        cleaned = []
        for item in urls:
            clean_url = _normalize_url_for_source_matching(item)
            if clean_url:
                cleaned.append(clean_url)
        return list(dict.fromkeys(cleaned))

    def _normalize_match_text(text):
        return re.sub(r"\s+", " ", str(text or "").strip()).lower()

    def _tokenize_for_block_matching(text):
        return re.findall(r"[a-z0-9]+", _normalize_match_text(text))

    def _token_f1_score(text_a, text_b):
        from collections import Counter

        tokens_a = _tokenize_for_block_matching(text_a)
        tokens_b = _tokenize_for_block_matching(text_b)
        if not tokens_a or not tokens_b:
            return 0.0

        counts_a = Counter(tokens_a)
        counts_b = Counter(tokens_b)
        overlap = sum(
            min(counts_a[token], counts_b[token])
            for token in (counts_a.keys() & counts_b.keys())
        )
        if overlap <= 0:
            return 0.0

        precision = overlap / len(tokens_a)
        recall = overlap / len(tokens_b)
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    raw_blocks_by_result_key = {}
    for result_key, replay_row in replay_payload.items():
        payload = replay_row.get(replay_mode) or {}
        response = payload.get("response") or {}
        blocks = _clean_response_text_blocks(
            _response_text_blocks_from_anthropic_response(response)
        )
        raw_blocks_by_result_key[result_key] = blocks

    def _best_matching_raw_claude_block(
        result_key,
        chunk_text,
        nli_citation_urls=None,
        min_fuzzy_score=0.45,
    ):
        blocks = raw_blocks_by_result_key.get(result_key, [])
        normalized_chunk = _normalize_match_text(chunk_text)
        normalized_nli_urls = {
            _normalize_url_for_source_matching(url)
            for url in (nli_citation_urls or [])
            if _normalize_url_for_source_matching(url)
        }
        if not normalized_chunk:
            return None
        for block in blocks:
            block_text = str(block.get("text", "") or "")
            if normalized_chunk in _normalize_match_text(block_text):
                return block

        best_block = None
        best_score = 0.0
        for block in blocks:
            score = _token_f1_score(chunk_text, block.get("text", ""))
            if score > best_score:
                best_score = score
                best_block = block
        if best_block is not None and best_score >= min_fuzzy_score:
            return best_block

        if normalized_nli_urls:
            best_url_block = None
            best_url_score = 0.0
            best_url_overlap = -1
            for block in blocks:
                block_urls = {
                    _normalize_url_for_source_matching(citation.get("url", ""))
                    for citation in (block.get("citations", []) or [])
                    if isinstance(citation, dict)
                    and _normalize_url_for_source_matching(citation.get("url", ""))
                }
                if not block_urls:
                    continue
                overlap = len(block_urls & normalized_nli_urls)
                if overlap <= 0:
                    continue
                score = _token_f1_score(chunk_text, block.get("text", ""))
                if (
                    overlap > best_url_overlap
                    or (overlap == best_url_overlap and score > best_url_score)
                ):
                    best_url_overlap = overlap
                    best_url_score = score
                    best_url_block = block
            if best_url_block is not None:
                return best_url_block
        return None

    df["nli_citation_urls"] = df["citation_urls"].apply(_normalize_url_list)
    df["raw_claude_matched_block"] = df.apply(
        lambda row: _best_matching_raw_claude_block(
            row.get("result_key"),
            row.get("response_chunk_text"),
            row.get("nli_citation_urls"),
        ),
        axis=1,
    )
    df["raw_claude_citation_urls"] = df["raw_claude_matched_block"].apply(
        lambda block: _normalize_url_list(
            [
                citation.get("url", "")
                for citation in ((block or {}).get("citations", []) or [])
                if isinstance(citation, dict)
            ]
        )
    )
    df["gold_has_associated_citation"] = df["raw_claude_citation_urls"].apply(bool)
    df["gold_has_null_citations"] = ~df["gold_has_associated_citation"]
    df["claude_bucket"] = np.where(
        df["gold_has_associated_citation"],
        "Associated Citations",
        "Parametric Knowledge",
    )
    df["predicted_bucket"] = (
        df["entailment_source_bucket"]
        .fillna("Parametric Knowledge")
        .replace(
            {
                "": "Parametric Knowledge",
                "Unknown": "Parametric Knowledge",
                "unknown": "Parametric Knowledge",
                "Unexplained": "Parametric Knowledge",
                "Marked Citations": "Associated Citations",
            }
        )
    )
    df["predicted_associated_citation"] = (
        df["predicted_bucket"] == "Associated Citations"
    )
    df["predicted_parametric_knowledge"] = (
        df["predicted_bucket"] == "Parametric Knowledge"
    )

    gold_df = df[df["gold_has_associated_citation"]].copy()
    if len(gold_df) == 0:
        print("No Claude-marked citation claims found in the replay NLI file.")
        return {}

    gold_df["entailed_url_normalized"] = gold_df["entailed_url"].apply(
        _normalize_url_for_source_matching
    )
    gold_df["entailed_url_in_claude_marked_urls"] = gold_df.apply(
        lambda row: row["entailed_url_normalized"] in set(row["raw_claude_citation_urls"]),
        axis=1,
    )
    gold_df["predicted_associated_and_same_url"] = (
        gold_df["predicted_associated_citation"]
        & gold_df["entailed_url_in_claude_marked_urls"]
    )
    parametric_df = df[df["predicted_parametric_knowledge"]].copy()
    parametric_df["claude_null_citations_match"] = parametric_df["gold_has_null_citations"]

    bucket_order = [
        "Associated Citations",
        "Other Citations",
        "Retrieved Sources",
        "Parametric Knowledge",
    ]
    bucket_counts = (
        gold_df["predicted_bucket"]
        .value_counts()
        .reindex(bucket_order, fill_value=0)
        .reset_index()
    )
    bucket_counts.columns = ["predicted_bucket", "count"]
    bucket_counts["percentage"] = bucket_counts["count"] / float(len(gold_df))
    pair_counts_df = (
        df.groupby(["claude_bucket", "predicted_bucket"])
        .size()
        .reset_index(name="count")
    )
    claude_bucket_order = ["Associated Citations", "Parametric Knowledge"]
    pair_counts_df["claude_bucket"] = pd.Categorical(
        pair_counts_df["claude_bucket"],
        categories=claude_bucket_order,
        ordered=True,
    )
    pair_counts_df["predicted_bucket"] = pd.Categorical(
        pair_counts_df["predicted_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    pair_counts_df = pair_counts_df.sort_values(
        ["claude_bucket", "predicted_bucket"]
    ).reset_index(drop=True)

    summary = {
        "model": model_name,
        "nli_method": nli_method,
        "chunking_method": chunking_method,
        "claim_selection_mode": claim_selection_mode,
        "input_path": str(input_path),
        "num_claude_marked_claims": int(len(gold_df)),
        "num_predicted_associated_citations": int(
            gold_df["predicted_associated_citation"].sum()
        ),
        "pct_predicted_associated_citations": float(
            gold_df["predicted_associated_citation"].mean()
        ),
        "num_predicted_associated_same_url": int(
            gold_df["predicted_associated_and_same_url"].sum()
        ),
        "pct_predicted_associated_same_url": float(
            gold_df["predicted_associated_and_same_url"].mean()
        ),
        "num_entailed_url_in_claude_marked_urls": int(
            gold_df["entailed_url_in_claude_marked_urls"].sum()
        ),
        "pct_entailed_url_in_claude_marked_urls": float(
            gold_df["entailed_url_in_claude_marked_urls"].mean()
        ),
        "num_predicted_parametric_claims": int(len(parametric_df)),
        "num_predicted_parametric_with_claude_null_citations": int(
            parametric_df["claude_null_citations_match"].sum()
        ),
        "pct_predicted_parametric_with_claude_null_citations": float(
            parametric_df["claude_null_citations_match"].mean()
        )
        if len(parametric_df) > 0
        else 0.0,
        "bucket_counts": bucket_counts.to_dict(orient="records"),
        "pair_counts": pair_counts_df.assign(
            claude_bucket=pair_counts_df["claude_bucket"].astype(str),
            predicted_bucket=pair_counts_df["predicted_bucket"].astype(str),
        ).to_dict(orient="records"),
    }

    bucket_counts.to_csv(
        output_dir
        / "replay_claude_associated_citation_bucket_alignment_summary.csv",
        index=False,
    )
    pair_counts_df.assign(
        claude_bucket=pair_counts_df["claude_bucket"].astype(str),
        predicted_bucket=pair_counts_df["predicted_bucket"].astype(str),
    ).to_csv(
        output_dir
        / "replay_claude_associated_citation_bucket_alignment_pair_counts.csv",
        index=False,
    )
    with open(
        output_dir
        / "replay_claude_associated_citation_bucket_alignment_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fig = go.Figure()
    color_map = {
        "Associated Citations": "#EF553B",
        "Other Citations": "#AB63FA",
        "Retrieved Sources": "#636EFA",
        "Parametric Knowledge": "#7F7F7F",
    }
    fig.add_trace(
        go.Bar(
            x=bucket_counts["predicted_bucket"],
            y=bucket_counts["percentage"],
            marker_color=[
                color_map.get(bucket, "#7F7F7F")
                for bucket in bucket_counts["predicted_bucket"]
            ],
            text=[
                f"{count}<br>{pct:.1%}"
                for count, pct in zip(
                    bucket_counts["count"],
                    bucket_counts["percentage"],
                )
            ],
            textposition="outside",
            customdata=bucket_counts["count"].to_numpy(),
            hovertemplate=(
                "Predicted bucket: %{x}<br>"
                "Claims: %{customdata}<br>"
                "Rate: %{y:.1%}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Predicted Source Bucket",
        yaxis_title="Rate of Claude-Marked Claims",
        title="Claude-Marked Claims by Predicted Source Bucket",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.write_html(
        output_dir / "replay_claude_associated_citation_bucket_alignment.html"
    )
    fig = with_paper_style(fig, config=styler(20, 16), legend_pos=None)
    fig.write_image(
        output_dir / "replay_claude_associated_citation_bucket_alignment.pdf",
        format="pdf",
    )

    print(
        f"Claude-marked claims: n={summary['num_claude_marked_claims']}; "
        f"predicted Associated Citations={summary['num_predicted_associated_citations']} "
        f"({summary['pct_predicted_associated_citations']:.1%}); "
        f"predicted Associated Citations with same entailed URL="
        f"{summary['num_predicted_associated_same_url']} "
        f"({summary['pct_predicted_associated_same_url']:.1%}); "
        f"predicted Parametric Knowledge={summary['num_predicted_parametric_claims']}, "
        f"of which Claude had null citations for "
        f"{summary['num_predicted_parametric_with_claude_null_citations']} "
        f"({summary['pct_predicted_parametric_with_claude_null_citations']:.1%})."
    )
    print("Judge bucket x Claude bucket claim counts:")
    for claude_bucket in claude_bucket_order:
        for predicted_bucket in bucket_order:
            pair_rows = pair_counts_df[
                (pair_counts_df["claude_bucket"].astype(str) == claude_bucket)
                & (pair_counts_df["predicted_bucket"].astype(str) == predicted_bucket)
            ]
            count = int(pair_rows["count"].iloc[0]) if len(pair_rows) > 0 else 0
            print(f"  Claude={claude_bucket} | Judge={predicted_bucket}: {count}")
    return summary


def evaluate_replay_source_tranco_ranks(
    ranked_input_path=Path(
        f"{OUTPUT_PATH}/replays/extracted/all_models_with_ranks_290726.json"
    ),
    separate_cited_external_internal=True,
    model_names=DEFAULT_MODELS,
    output_dir=PLOT_OUTPUT_DIR / "source_selection",
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from src.utils.figure_style import with_paper_style, styler

    ranked_payload = _load_replay_json(ranked_input_path)

    def _flatten_rank_values(value):
        flat = []
        if not isinstance(value, list):
            return flat
        for item in value:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    def _avg_valid_rank(ranks):
        valid_ranks = []
        for rank in _flatten_rank_values(ranks):
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
        valid_ranks = []
        flat_ranks = _flatten_rank_values(ranks)
        for idx, rank in enumerate(flat_ranks):
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

    replay_rows = _extract_rows_for_models(model_names, tool_choice="auto")
    web_call_lookup = {
        (row.get("model"), row.get("result_key")): bool(row.get("has_web_tool_call", False))
        for row in replay_rows
        if row.get("result_key")
    }

    model_frames = {}
    for model_name in model_names:
        model_records = []
        for result_key, sample in ranked_payload.items():
            models_payload = sample.get("models", {})
            model_payload = models_payload.get(model_name, {})
            if not isinstance(model_payload, dict):
                continue
            if not bool(web_call_lookup.get((model_name, result_key), False)):
                continue
            model_records.append(
                {
                    "result_key": result_key,
                    "conv_id": sample.get("conv_id"),
                    "turn_id": sample.get("turn_id"),
                    "sources_retrieved": model_payload.get("sources_retrieved", []),
                    "sources_cited": model_payload.get("sources_cited", []),
                    "ranks_srcs_retrieved": model_payload.get(
                        "sources_retrieved_ranks", []
                    ),
                    "ranks_srcs_cited": model_payload.get("sources_cited_ranks", []),
                }
            )
        model_frames[model_name] = pd.DataFrame(model_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for model_name in model_names:
        df = model_frames.get(model_name)
        if df is None or len(df) == 0:
            print(f"{_model_label(model_name)}: no ranked replay rows.")
            continue

        df["retrieved_avg_rank"] = df["ranks_srcs_retrieved"].apply(_avg_valid_rank)

        if separate_cited_external_internal:
            cited_external_avg = []
            cited_internal_avg = []
            for _, row in df.iterrows():
                retrieved_urls = {
                    _normalize_url_for_source_matching(item.get("url", ""))
                    for item in _flatten_source_items(row.get("sources_retrieved", []))
                    if isinstance(item, dict) and item.get("url", "")
                }
                cited_sources = _flatten_source_items(row.get("sources_cited", []))
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

                cited_external_avg.append(
                    _avg_valid_rank_by_mask(
                        row.get("ranks_srcs_cited", []), external_mask
                    )
                )
                cited_internal_avg.append(
                    _avg_valid_rank_by_mask(
                        row.get("ranks_srcs_cited", []), internal_mask
                    )
                )

            df["cited_external_avg_rank"] = cited_external_avg
            df["cited_internal_avg_rank"] = cited_internal_avg
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

        print(f"{_model_label(model_name)} average Tranco rank ranges:")
        summary[model_name] = {"ranges": {}}
        for col, label, _color in rank_specs:
            subset = df[col].dropna()
            if len(subset) == 0:
                print(f"{label}: no valid ranks")
                summary[model_name]["ranges"][label] = None
            else:
                print(f"{label}: {subset.min():,.0f}-{subset.max():,.0f}")
                summary[model_name]["ranges"][label] = {
                    "min": float(subset.min()),
                    "max": float(subset.max()),
                }

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
            yaxis=dict(
                range=[global_min_log, global_max_log],
                tickmode="linear",
                dtick=1,
            ),
            violinmode="group",
            margin=dict(t=5, b=130, r=5),
        )
        model_slug = str(model_name).replace(".", "-")
        violin_file_name = f"replay_source_rank_violinplot__{model_slug}"
        if separate_cited_external_internal:
            violin_file_name += "_split_cited"
        box_fig = with_paper_style(box_fig, config=styler(26, 16), legend_pos=None)
        box_fig.update_xaxes(tickangle=0, tickfont=dict(size=26))
        box_fig.write_image(output_dir / f"{violin_file_name}.pdf", format="pdf")

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

        paired_file_name = f"replay_source_rank_paired_plot__{model_slug}"
        if separate_cited_external_internal:
            paired_file_name += "_split_cited"
        legend_pos = (0.98, 1.2) if separate_cited_external_internal else None
        paired_fig = with_paper_style(
            paired_fig,
            config=styler(18, 16),
            legend_pos=legend_pos,
        )
        paired_fig.update_layout(
            width=700 if separate_cited_external_internal else 500,
            height=400,
        )
        paired_fig.write_image(output_dir / f"{paired_file_name}.pdf", format="pdf")

    with open(
        output_dir
        / (
            "replay_source_tranco_rank_summary_split_cited.json"
            if separate_cited_external_internal
            else "replay_source_tranco_rank_summary.json"
        ),
        "w",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def print_replay_web_search_sample_counts(
    model_groups=None,
    base_model_name="gpt-5.3-chat-latest",
):
    if model_groups is None:
        model_groups = {
            "openai_models": OPENAI_REPLAY_MODELS,
            "cross_platform_models": DEFAULT_MODELS,
        }

    for group_name, model_names in model_groups.items():
        rows = _extract_rows_for_models(model_names)
        grouped_samples = _group_rows_by_sample(rows)
        print(f"{group_name}: {len(grouped_samples)} samples")

        for model_name in model_names:
            model_rows = [row for row in rows if row.get("model") == model_name]
            called_count = sum(
                1
                for sample in grouped_samples.values()
                if bool(sample["models"].get(model_name, False))
            )
            print(f"  {_model_label(model_name)}: {called_count}")

            iteration_counts = {"1": 0, "2": 0, "3+": 0}
            for row in model_rows:
                web_query_groups = _clean_web_query_groups(row.get("web_queries", []))
                if not web_query_groups:
                    continue
                num_iterations = len(web_query_groups)
                if num_iterations <= 1:
                    iteration_counts["1"] += 1
                elif num_iterations == 2:
                    iteration_counts["2"] += 1
                else:
                    iteration_counts["3+"] += 1
            print(
                "    "
                f"iterations: 1={iteration_counts['1']}, "
                f"2={iteration_counts['2']}, "
                f"3+={iteration_counts['3+']}"
            )

        if base_model_name in model_names:
            base_label = _model_label(base_model_name)
            for model_name in model_names:
                if model_name == base_model_name:
                    continue

                both_called = 0
                base_only = 0
                neither_called = 0
                model_only = 0

                for sample in grouped_samples.values():
                    base_called = bool(sample["models"].get(base_model_name, False))
                    model_called = bool(sample["models"].get(model_name, False))
                    if base_called and model_called:
                        both_called += 1
                    elif base_called and not model_called:
                        base_only += 1
                    elif not base_called and not model_called:
                        neither_called += 1
                    else:
                        model_only += 1

                print(
                    "  "
                    f"{_model_label(model_name)} vs {base_label}: "
                    f"both_called={both_called}, "
                    f"{base_label}_only={base_only}, "
                    f"neither={neither_called}, "
                    f"model_only={model_only}"
                )


def classify_replay_sample_primary_triggers(
    model_names=DEFAULT_MODELS,
    replay_path=REPLAY_SAMPLE_SOURCE_PATH,
    output_path=REPLAY_SAMPLE_CHARACTERIZATION_PATH,
    save_every=25,
):
    from src.replays.chat_replayer import _client_for_provider, _infer_provider_from_model

    replay_data = _load_replay_json(replay_path)
    if isinstance(model_names, str):
        model_names = [model_names]
    model_names = list(dict.fromkeys(model_names))

    samples = []
    for row in replay_data.values():
        if _is_skipped_sample_idx(row.get("sample_idx")):
            continue
        samples.append(
            {
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "user_prompt": row.get("user_prompt", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    completed_keys = set()
    clients = {
        model_name: _client_for_provider(_infer_provider_from_model(model_name))
        for model_name in model_names
    }

    def _save():
        fieldnames = [
            "conv_id",
            "turn_id",
            "user_prompt",
            "judge_model",
            "followed_web_policy",
            "primary_trigger",
            "secondary_triggers",
            "explanation",
            "judge_status",
            "judge_error",
            "judge_raw_judgment",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    f'{row.get("conv_id")}::{row.get("turn_id")}::{row.get("judge_model")}': row
                    for row in records
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    for model_name in model_names:
        client = clients[model_name]
        for row in tqdm(samples, desc=f"judge={model_name}"):
            record_key = (
                str(row.get("conv_id", "") or ""),
                str(row.get("turn_id", "") or ""),
                str(model_name),
            )
            if record_key in completed_keys:
                continue

            user_prompt = str(row.get("user_prompt", "") or "").strip()
            record = {
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "user_prompt": user_prompt,
                "judge_model": model_name,
                "followed_web_policy": "",
                "primary_trigger": "",
                "secondary_triggers": json.dumps([], ensure_ascii=False),
                "explanation": "",
                "judge_status": "",
                "judge_error": "",
                "judge_raw_judgment": "",
            }

            try:
                eval_result = _run_judge(
                    client=client,
                    model_name=model_name,
                    system_prompt=SYSTEM_PROMPT_CHARAC,
                    user_prompt=USER_PROMPT_CHARAC.format(PROMPT=user_prompt),
                )
                parsed = eval_result.get("parsed_judgment")
                if not isinstance(parsed, dict):
                    parsed = {}

                record["followed_web_policy"] = json.dumps(
                    parsed,
                    ensure_ascii=False,
                )
                record["primary_trigger"] = parsed.get("primary_trigger", "")
                record["secondary_triggers"] = json.dumps(
                    parsed.get("secondary_triggers", []),
                    ensure_ascii=False,
                )
                record["explanation"] = parsed.get("explanation", "")
                record["judge_status"] = "ok" if parsed else "parse_failed"
                record["judge_raw_judgment"] = eval_result.get("raw_judgment", "")
            except Exception as exc:
                record["judge_status"] = "error"
                record["judge_error"] = str(exc)

            records.append(record)
            completed_keys.add(record_key)
            if save_every and len(records) % save_every == 0:
                _save()

    _save()


def data_extraction():
    all_rows = []
    for model_name in DEFAULT_MODELS:
        print(model_name)
        rows = extract_model_file(model_name, INPUT_DIR)
        print(f"kept rows: {len(rows)}")
        all_rows.extend(rows)
        save_outputs(rows, OUTPUT_DIR / model_name)
        # save_sample_web_queries(rows, OUTPUT_DIR / f"{model_name}__web_queries")

    combined_output = save_outputs(all_rows, OUTPUT_DIR / "all_models")
    save_sample_web_queries(all_rows, OUTPUT_DIR / "all_models__web_queries")
    save_prompt_and_model_web_queries(
        all_rows,
        OUTPUT_DIR / "all_models__prompt_and_web_queries.json",
    )
    print(f"total rows: {len(all_rows)}")
    if isinstance(combined_output, list):
        print("pandas not available; wrote JSON and CSV only.")


if __name__ == "__main__":
    # data_extraction()
    
    # classify_replay_sample_primary_triggers()
    # plot_openai_replay_model_outcome_trigger_heatmaps()
    # plot_replay_pair_outcome_trigger_heatmap(
    #     # base_model_name="gpt-4.1-mini-2025-04-14",
    #     base_model_name="gpt-5.3-chat-latest",
    #     comparison_model_name="o4-mini-2025-04-16",
    #     # comparison_model_name="gpt-4.1-mini-2025-04-14",
    # )
    
    # plot_replay_web_call_agreement_counts()
    # plot_openai_replay_model_agreement_counts()

    # plot_openai_replay_model_call_outcomes()

    # plot_cross_platform_replay_model_call_outcomes()

    # print_replay_web_search_sample_counts()

    # plot_replay_query_term_count_trends_over_time()

    # plot_replay_parallel_queries_by_query_reformulations()
    # plot_replay_parallel_queries_by_query_reformulations(model_names=OPENAI_REPLAY_MODELS)

    # plot_replay_top_domains(
    #     common_samples_only=True, 
    #     common_model_names=[
    #         "gpt-5.3-chat-latest",
    #         "claude-sonnet-4-6",
    #         "grok-4.3",
    #         "deepseek-v4-flash"
    #     ],
    # )
    # plot_replay_top_domains(
    #     common_samples_only=False,
    #     common_model_names=[
    #         "gpt-5.3-chat-latest",
    #         "claude-sonnet-4-6",
    #         "grok-4.3",
    #         "deepseek-v4-flash"
    #     ],
    # )

    # print_cross_platform_replay_model_call_outcome_eval_scores()

    # query_specificity_evaluation()
    # plot_query_specificity_distribution_by_iteration()

    # reasons_for_another_web_query()
    # plot_reasons_for_another_web_query_distribution_all_models()

    # compute_average_citations_and_retrievals_per_response_for_replays()

    # plot_openai_replay_dev_prompt_web_call_heatmap()

    # evaluate_replay_source_tranco_ranks(separate_cited_external_internal=False)
    # evaluate_replay_source_tranco_ranks(separate_cited_external_internal=True)

    # asyncio.run(
    #     extract_replay_urls_content(
    #         # model_names=["grok-4.3"],
    #         common_filter_model_names=[
    #             "grok-4.3",
    #             "claude-sonnet-4-6",
    #             "gpt-5.3-chat-latest",
    #         ],
    #         # model_names=["gpt-5.3-chat-latest"],
    #         model_names=["claude-sonnet-4-6"],
    #         replay_mode="auto",
    #         force_refresh=False,
    #     )
    # )

    # response_source_nli_sentence_based_for_replays(
    #     # model_names=["grok-4.3"],
    #     model_names=["gpt-5.3-chat-latest"],
    #     # model_names=["claude-sonnet-4-6"],
    #     common_filter_model_names=[
    #         "grok-4.3",
    #         "claude-sonnet-4-6",
    #         "gpt-5.3-chat-latest",
    #     ],
    #     nli_method="judge",
    #     # nli_method="bert",
    #     chunking_method="claim",
    #     claim_selection_mode="all",
    #     # claim_selection_mode="latest_preceding",
    # )

    # plot_response_source_nli_sentence_based_judge_for_replays(
    #     model_names=[
    #         "gpt-5.3-chat-latest",
    #         "claude-sonnet-4-6",
    #         "grok-4.3",
    #     ],
    #     common_filter_model_names=[
    #         "gpt-5.3-chat-latest",
    #         "claude-sonnet-4-6",
    #         "grok-4.3",
    #     ],
    # )

    # evaluate_claude_associated_citation_bucket_alignment_for_replays()

    # response_source_nli_sentence_based_factuality_for_replays(
    #     model_names=DEFAULT_MODELS,
    #     nli_method="judge",
    #     chunking_method="claim",
    #     claim_selection_mode="all",
    # )

    # summarize_response_source_nli_sentence_based_factuality_for_replays(
    #     model_names=DEFAULT_MODELS,
    #     nli_method="judge",
    #     chunking_method="claim",
    #     claim_selection_mode="all",
    # )
    pass
