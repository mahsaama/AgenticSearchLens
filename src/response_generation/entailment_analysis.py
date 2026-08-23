"""Scores whether a source's text entails, contradicts, or is neutral to a
claim/response (NLI), and the factuality/grounding-source analysis built on
top of that scoring: per-claim LLM-judge and BERT NLI comparisons, bucketed
confidence intervals, and the plots/tables for §5.2 of the paper.

compute_nli_scores(premise, hypothesis) is the core primitive: one LLM call
(SYSTEM_PROMPT_RESP_SYNT) scoring a single source/response pair.
response_source_nli_sentence_based(...) is the main entry point built on
top of it -- for every sampled turn (see web_content_fetch.py's
_load_response_source_similarity_input()), extracts claims from the
response (via claim_extraction.extract_claims_from_text(), cached), fetches
the matching source text (via web_content_fetch._load_urls_content()), and
scores each claim/source pair. evaluate_claim_factuality(...) is the
parallel LLM-judge path used to score factuality by grounding-source
bucket (associated citation / other citation / search result / parametric
knowledge) rather than plain entailment.

Split out of response_generation.py, alongside web_content_fetch.py and
claim_extraction.py -- response_generation.py imports the entry points it
needs from all three rather than defining them itself.

Run directly (`python -m src.response_generation.entailment_analysis`) for
a small smoke test: scores one hardcoded entailing and one hardcoded
contradicting premise/hypothesis pair and prints the judgments (needs
OPENAI_API_KEY; doesn't touch outputs/).
"""

import ast
import json
import logging
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.prompts.evaluator_prompts import (
    SYSTEM_PROMPT_CLAIM_FACTUALITY_EVAL,
    SYSTEM_PROMPT_RESP_SYNT,
    USER_PROMPT_CLAIM_FACTUALITY_EVAL,
    USER_PROMPT_RESP_SYNT,
)
from src.utils.common_io import OUTPUT_PATH, load_json, to_json
from src.utils.figure_style import with_paper_style, styler
from src.response_generation.claim_extraction import (
    CLAIM_EXTRACTION_CACHE_PATH,
    _build_claim_cache_user_query_lookup,
    _claim_cache_claim_texts,
    _claim_cache_factuality_output_path,
    _claim_cache_key,
    _load_claims_cache,
    _row_user_query,
    _save_claims_cache,
    extract_claims_from_text,
    extract_first_json_object,
)
from src.response_generation.web_content_fetch import (
    RESPONSE_URLS_CONTENT_PATH,
    _load_response_source_similarity_input,
    _load_urls_content,
)

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

pio.defaults.mathjax = None
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CONF = "./response_generation"

RESPONSE_SOURCE_EFFECT_EVALUATIONS_BASE = (
    f"{OUTPUT_PATH}/chatgpt/metadata/response_source_effect_evaluations"
)
RESPONSE_SOURCE_NLI_SENTENCE_BASED_JUDGE_BASE = (
    f"{OUTPUT_PATH}/chatgpt/metadata/response_source_nli_sentence_based_judge"
)
RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE = (
    f"{OUTPUT_PATH}/chatgpt/metadata/response_source_nli_sentence_based_bert"
)
EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES = {
    "Claude": {
        "bert": (
            f"{OUTPUT_PATH}/claude/metadata/response_source_nli_sentence_based_bert_claim_latest_preceding"
        ),
        "judge": (
            f"{OUTPUT_PATH}/claude/metadata/response_source_nli_sentence_based_judge_claim_latest_preceding"
        ),
    },
    "Grok": {
        "bert": (
            f"{OUTPUT_PATH}/grok/metadata/response_source_nli_sentence_based_bert_claim_latest_preceding"
        ),
        "judge": (
            f"{OUTPUT_PATH}/grok/metadata/response_source_nli_sentence_based_judge_claim_latest_preceding"
        ),
    },
    "DeepSeek": {
        "bert": (
            f"{OUTPUT_PATH}/deepseek/metadata/response_source_nli_sentence_based_bert_claim_latest_preceding"
        ),
        "judge": (
            f"{OUTPUT_PATH}/deepseek/metadata/response_source_nli_sentence_based_judge_claim_latest_preceding"
        ),
    },
}
EXTERNAL_PLATFORM_ORDER = ["ChatGPT"] + list(
    EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys()
)

CITED_URL_VALIDITY_LABELS_PATH = (
    f"{OUTPUT_PATH}/chatgpt/metadata/cited_url_validity_labels.json"
)
BERT_NLI_MODEL_NAME = os.getenv("BERT_NLI_MODEL_NAME")
BERT_NLI_MAX_LENGTH = int(os.getenv("BERT_NLI_MAX_LENGTH"))
NLI_JUDGE_CONTEXT_WINDOW_TOKENS = 128000
NLI_JUDGE_MAX_OUTPUT_TOKENS = 256
NLI_JUDGE_TOKEN_SAFETY_MARGIN = 2000
NLI_ESTIMATED_CHARS_PER_TOKEN = 3.0

FACTUALITY_JUDGE_MODEL = os.getenv("FACTUALITY_JUDGE_MODEL")
FACTUALITY_JUDGE_MAX_OUTPUT_TOKENS = int(
    os.getenv("FACTUALITY_JUDGE_MAX_OUTPUT_TOKENS")
)

def _record_join_key(record):
    return tuple(
        str(record.get(col, "") or "").strip()
        for col in ["user_id", "conv_id", "turn_id"]
    )


def _build_response_source_user_query_lookup(platform="chatgpt"):
    response_df = _load_response_source_similarity_input(platform=platform).copy()
    user_query_lookup = {}

    for _, row in response_df.iterrows():
        record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        join_key = _record_join_key(record)
        if not any(join_key):
            continue
        user_query = _row_user_query(record)
        if user_query:
            user_query_lookup[join_key] = user_query

    return user_query_lookup


def _safe_parse_source_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    return parsed if isinstance(parsed, list) else []


def _external_platform_output_base_candidates(
    platform_output_base,
    source_text_mode="full_url_content",
):
    if not platform_output_base:
        return []

    source_text_mode = _normalize_source_text_mode(source_text_mode)
    candidates = []

    def _append_candidate(value):
        if value and value not in candidates:
            candidates.append(value)

    base_candidates = [platform_output_base]
    if platform_output_base.endswith("_latest_preceding"):
        base_candidates.append(platform_output_base.removesuffix("_latest_preceding"))

    if source_text_mode != "full_url_content":
        for base_value in base_candidates:
            _append_candidate(f"{base_value}_{source_text_mode}")
        return candidates

    for base_value in base_candidates:
        _append_candidate(base_value)

    return candidates

def _estimate_token_count(text):
    if not text:
        return 0
    return int(np.ceil(len(text) / NLI_ESTIMATED_CHARS_PER_TOKEN))

def _trim_nli_source_to_context(source_text, response_text):
    source_text = str(source_text or "")
    response_text = str(response_text or "")
    if not source_text:
        return source_text

    base_prompt_text = (
        str(SYSTEM_PROMPT_RESP_SYNT or "")
        + USER_PROMPT_RESP_SYNT.format(response_text=response_text, source="")
    )
    base_tokens = _estimate_token_count(base_prompt_text)
    source_token_budget = (
        NLI_JUDGE_CONTEXT_WINDOW_TOKENS
        - NLI_JUDGE_MAX_OUTPUT_TOKENS
        - NLI_JUDGE_TOKEN_SAFETY_MARGIN
        - base_tokens
    )
    if source_token_budget <= 0:
        return ""

    source_tokens = _estimate_token_count(source_text)
    if source_tokens <= source_token_budget:
        return source_text

    max_source_chars = int(source_token_budget * NLI_ESTIMATED_CHARS_PER_TOKEN)
    if max_source_chars <= 0:
        return ""
    return source_text[:max_source_chars]

def compute_nli_scores(premise, hypothesis):
    """Score whether a source text entails, contradicts, or is neutral to the response."""
    premise = _trim_nli_source_to_context(premise, hypothesis).strip()
    hypothesis = str(hypothesis or "").strip()

    msg = [
        {"role": "system", "content": SYSTEM_PROMPT_RESP_SYNT},
        {
            "role": "user",
            "content": USER_PROMPT_RESP_SYNT.format(
                response_text=hypothesis,
                source=premise,
            ),
        },
    ]
    text = ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msg,
            max_tokens=256,
            temperature=0.0,
        )
        text = response.choices[0].message.content
    except Exception as e:
        print(e)
        text = ""

    json_response = extract_first_json_object(text)
    return json_response

def _load_response_source_nli_sentence_based(output_base=None):
    output_base = output_base or RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE
    pkl_path = f"{output_base}.pkl"
    csv_path = f"{output_base}.csv"
    json_path = f"{output_base}.json"

    if os.path.exists(pkl_path):
        try:
            return pd.read_pickle(pkl_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", pkl_path, e)

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    if os.path.exists(json_path):
        records = load_json(json_path)
        if isinstance(records, list):
            return pd.DataFrame(records)

    raise FileNotFoundError(
        f"Sentence-level response source NLI results not found at {output_base}.*. "
        "Run response_source_nli_sentence_based() first."
    )


def _load_response_source_nli_sentence_based_records(input_path=None, output_base=None):
    if input_path:
        records = load_json(input_path)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list at {input_path}")
        return records, input_path

    output_base = output_base or RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE
    json_path = f"{output_base}.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Sentence-level response source NLI JSON results not found at {json_path}."
        )

    records = load_json(json_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list at {json_path}")
    return records, json_path


def _coerce_factuality_payload(payload):
    payload = payload if isinstance(payload, dict) else {}
    score = payload.get("score", payload.get("rating", 0))
    try:
        score = float(score or 0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "label": str(payload.get("label", payload.get("verdict", "")) or "").strip(),
        "score": score,
        "reasoning": str(
            payload.get("reasoning", payload.get("reason", "")) or ""
        ).strip(),
    }


def _format_prompt(template, **kwargs):
    try:
        return template.format(**kwargs)
    except KeyError:
        reduced_kwargs = {
            key: value for key, value in kwargs.items() if key in {"claim", "user_query"}
        }
        try:
            return template.format(**reduced_kwargs)
        except KeyError:
            if "claim" in kwargs:
                return template.format(claim=kwargs.get("claim", ""))
            return template


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
    if isinstance(value, pd.Timestamp):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def evaluate_claim_factuality(
    claim,
    user_query="",
    model_name=FACTUALITY_JUDGE_MODEL,
):
    claim = str(claim or "").strip()
    user_query = str(user_query or "").strip()
    if not claim:
        return {
            "label": "",
            "score": 0.0,
            "reasoning": "",
            "raw_response": "",
            "model": model_name,
        }

    msg = [
        {"role": "system", "content": SYSTEM_PROMPT_CLAIM_FACTUALITY_EVAL},
        {
            "role": "user",
            "content": _format_prompt(
                USER_PROMPT_CLAIM_FACTUALITY_EVAL,
                user_query=user_query,
                claim=claim,
            ),
        },
    ]

    response_text = ""
    try:
        response = client.responses.create(
            model=model_name,
            tools=[{"type": "web_search"}],
            tool_choice="required",
            input=msg,
            max_output_tokens=FACTUALITY_JUDGE_MAX_OUTPUT_TOKENS,
        )
        response_text = response.output_text or ""
    except Exception as e:
        logger.warning("Claim factuality evaluation failed: %s", e)
        return {
            "label": "",
            "score": 0.0,
            "reasoning": "",
            "raw_response": str(e) if not response_text else response_text,
            "model": model_name,
        }

    parsed_payload = None
    try:
        parsed_payload = json.loads(response_text)
    except Exception:
        parsed_payload = extract_first_json_object(response_text)

    factuality_payload = _coerce_factuality_payload(parsed_payload)
    factuality_payload["raw_response"] = response_text
    factuality_payload["model"] = model_name
    return factuality_payload


def _normalize_chunking_method(chunking_method):
    chunking_method_map = {
        "citation_marker": "citation_marker",
        "marker": "citation_marker",
        "citation": "citation_marker",
        "chunk": "citation_marker",
        "chunk_based": "citation_marker",
        "claim": "claim",
        "claim_based": "claim",
        "sentence": "claim",
        "sentence_based": "claim",
    }
    chunking_method_key = str(chunking_method or "").strip().lower()
    if chunking_method_key not in chunking_method_map:
        raise ValueError(
            "chunking_method must be one of {'citation_marker', 'claim'}"
        )
    return chunking_method_map[chunking_method_key]


def _normalize_claim_selection_mode(claim_selection_mode):
    claim_selection_mode_map = {
        "all": "all",
        "all_claims": "all",
        "all_claims_in_chunk": "all",
        "latest_preceding": "latest_preceding",
        "latest_preceding_claim": "latest_preceding",
        "latest_before_marker": "latest_preceding",
        "immediate_predecessor": "latest_preceding",
    }
    mode_key = str(claim_selection_mode or "").strip().lower()
    if mode_key not in claim_selection_mode_map:
        raise ValueError(
            "claim_selection_mode must be one of {'all', 'latest_preceding'}"
        )
    return claim_selection_mode_map[mode_key]


def _normalize_source_text_mode(source_text_mode):
    source_text_mode_map = {
        "full_url_content": "full_url_content",
        "full": "full_url_content",
        "page_content": "full_url_content",
        "url_content": "full_url_content",
        "snippet": "snippet",
        "snippets": "snippet",
    }
    mode_key = str(source_text_mode or "").strip().lower()
    if mode_key not in source_text_mode_map:
        raise ValueError(
            "source_text_mode must be one of {'full_url_content', 'snippet'}"
        )
    return source_text_mode_map[mode_key]


def _response_source_nli_output_base(
    nli_method,
    chunking_method,
    claim_selection_mode="all",
    source_text_mode="full_url_content",
    platform="chatgpt",
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    source_text_mode = _normalize_source_text_mode(source_text_mode)
    method_base = f"{OUTPUT_PATH}/{platform}/metadata/response_source_nli_sentence_based_{nli_method}"
    if chunking_method == "citation_marker":
        output_base = method_base
    elif claim_selection_mode == "all":
        output_base = f"{method_base}_{chunking_method}"
    else:
        output_base = f"{method_base}_{chunking_method}_{claim_selection_mode}"
    if source_text_mode != "full_url_content":
        output_base = f"{output_base}_{source_text_mode}"
    return output_base


def response_source_nli_sentence_based(
    nli_method="bert",
    judge_entailment_min_score=1,
    chunking_method="citation_marker",
    claim_selection_mode="latest_preceding",
    source_text_mode="full_url_content",
    claim_cache_path=None,
    platform="chatgpt",
):
    """Attribute response chunks using either judge NLI or BERT NLI."""
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    source_text_mode = _normalize_source_text_mode(source_text_mode)
    output_base = _response_source_nli_output_base(
        nli_method,
        chunking_method,
        claim_selection_mode=claim_selection_mode,
        source_text_mode=source_text_mode,
        platform=platform,
    )
    if claim_cache_path is None:
        claim_cache_path = (
            CLAIM_EXTRACTION_CACHE_PATH
            if platform == "chatgpt"
            else f"{OUTPUT_PATH}/{platform}/metadata/response_source_claim_chunks_cache.json"
        )
    persisted_claims_cache = _load_claims_cache(cache_path=claim_cache_path)
    claims_cache_dirty = False
    new_claim_cache_entries = 0

    df = _load_response_source_similarity_input(platform=platform)
    urls_content_by_clean_url = {}

    urls_content_path = (
        RESPONSE_URLS_CONTENT_PATH
        if platform == "chatgpt"
        else f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_url_content.json"
    )
    urls_content = _load_urls_content(
        urls_content_path=urls_content_path,
        required=False,
    )
    print(len(urls_content.keys()))
    for url, content in urls_content.items():
        clean_url = str(url).removesuffix("?utm_source=chatgpt.com").removesuffix(
            "&utm_source=chatgpt.com"
        )
        urls_content_by_clean_url[clean_url] = content
        urls_content_by_clean_url[clean_url.rstrip("/")] = content

    citation_marker_pattern = re.compile(
        r"\ue200(?=[^\ue201]*\ue202[A-Za-z]+\d+[A-Za-z]+\d+(?:\ue202|\ue201))[^\ue201]*\ue201"
    )
    citation_ref_pattern = re.compile(
        r"\ue202[A-Za-z]+(\d+)[A-Za-z]+(\d+)(?=\ue202|\ue201)"
    )

    def _safe_int(value):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _clean_url(url):
        if not isinstance(url, str):
            return ""
        return url.strip().removesuffix("?utm_source=chatgpt.com").removesuffix(
            "&utm_source=chatgpt.com"
        )

    def _as_source_list(value):
        if isinstance(value, list):
            return value
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
        return parsed if isinstance(parsed, list) else []

    claims_cache = {}

    def _extract_claims(text):
        nonlocal claims_cache_dirty
        nonlocal new_claim_cache_entries
        text = str(text or "").strip()
        if not text:
            return []
        if text in claims_cache:
            return claims_cache[text]

        cache_key = _claim_cache_key(text)
        cached_entries = persisted_claims_cache.get(cache_key, []) or persisted_claims_cache.get(text, [])
        cached_claims = _claim_cache_claim_texts(cached_entries)
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

    def _split_sentences(text):
        if not text:
            return []
        if chunking_method == "claim":
            sentence_parts = _extract_claims(text)
        else:
            sentence_parts = re.split(
                r"(?<=[.!?])\s+(?=[`'*_\"(\[]*[A-Z0-9])",
                text,
            )

        sentences = []
        for sentence in sentence_parts:
            sentence = sentence.strip(" -*\t\n")
            if len(sentence) < 8 or not re.search(r"[A-Za-z]", sentence):
                continue
            sentences.append(sentence)
        return sentences

    def _citation_refs(marker_text):
        refs = []
        for turn_index, ref_index in citation_ref_pattern.findall(marker_text or ""):
            refs.append((int(turn_index), int(ref_index)))
        return refs

    def _append_response_chunks(rows, sentences, citation_refs, citation_markers):
        citation_refs = list(citation_refs or [])
        citation_markers = list(citation_markers or [])
        if citation_refs and chunking_method == "citation_marker":
            if not sentences:
                return
            rows.append(
                {
                    "sentence": " ".join(sentences),
                    "sentences": sentences,
                    "sentence_count": len(sentences),
                    "citation_refs": list(citation_refs),
                    "citation_markers": list(citation_markers),
                }
            )
            return

        for sentence in sentences:
            rows.append(
                {
                    "sentence": sentence,
                    "sentences": [sentence],
                    "sentence_count": 1,
                    "citation_refs": list(citation_refs),
                    "citation_markers": list(citation_markers),
                }
            )

    def _extract_response_sentences(response_text):
        rows = []
        response_text = response_text or ""
        marker_matches = list(citation_marker_pattern.finditer(response_text))

        if (
            chunking_method == "claim"
            and claim_selection_mode == "latest_preceding"
        ):
            previous_end = 0
            last_refs_key = None
            for marker_match in marker_matches:
                marker_text = marker_match.group(0)
                marker_refs = _citation_refs(marker_text)
                refs_key = tuple(marker_refs)
                predecessor_claims = _split_sentences(
                    response_text[previous_end:marker_match.start()]
                )

                if not marker_refs:
                    previous_end = marker_match.end()
                    last_refs_key = None
                    continue

                if predecessor_claims:
                    latest_claim = predecessor_claims[-1]
                    # For repeated markers with the same refs, keep only the latest
                    # preceding claim in the current marker chunk.
                    if rows and refs_key == last_refs_key:
                        rows[-1]["sentence"] = latest_claim
                        rows[-1]["sentences"] = [latest_claim]
                        rows[-1]["sentence_count"] = 1
                        rows[-1]["citation_markers"].append(marker_text)
                    else:
                        _append_response_chunks(
                            rows,
                            [latest_claim],
                            marker_refs,
                            [marker_text],
                        )
                elif rows and refs_key == last_refs_key:
                    # Keep marker metadata even when no new predecessor claim appears.
                    rows[-1]["citation_markers"].append(marker_text)

                if marker_refs:
                    last_refs_key = refs_key

                previous_end = marker_match.end()

            return rows

        previous_end = 0
        for marker_match in marker_matches:
            marker_text = marker_match.group(0)
            marker_refs = _citation_refs(marker_text)
            raw_chunk = response_text[previous_end:marker_match.start()]
            chunk_sentences = _split_sentences(raw_chunk)

            if not chunk_sentences and rows and marker_refs:
                rows[-1]["citation_refs"].extend(
                    ref for ref in marker_refs if ref not in rows[-1]["citation_refs"]
                )
                rows[-1]["citation_markers"].append(marker_text)
            else:
                _append_response_chunks(
                    rows,
                    chunk_sentences,
                    marker_refs,
                    [marker_text],
                )

            previous_end = marker_match.end()

        tail_sentences = _split_sentences(response_text[previous_end:])
        _append_response_chunks(rows, tail_sentences, [], [])

        if not rows:
            _append_response_chunks(rows, _split_sentences(response_text), [], [])

        return rows

    def _source_records(row):
        records = []
        for source_col, source_type in [
            ("srcs_cited", "Cited"),
            ("srcs_retrieved", "Retrieved"),
        ]:
            for src in _as_source_list(row.get(source_col, [])):
                if not isinstance(src, dict):
                    continue
                url = _clean_url(src.get("url", ""))
                if not url:
                    continue
                turn_index = _safe_int(src.get("turn_index"))
                ref_index = _safe_int(src.get("ref_index"))
                records.append(
                    {
                        "url": url,
                        "source_type": source_type,
                        "turn_index": turn_index,
                        "ref_index": ref_index,
                        "ref_key": (
                            (turn_index, ref_index)
                            if turn_index is not None and ref_index is not None
                            else None
                        ),
                        "domain": src.get("domain", ""),
                        "title": src.get("title", ""),
                        "snippet": src.get("snippet", ""),
                    }
                )
        return records

    def _source_content(url):
        url = _clean_url(url)
        return str(
            urls_content_by_clean_url.get(
                url,
                urls_content_by_clean_url.get(url.rstrip("/"), ""),
            )
            or ""
        )

    def _source_text(source):
        source = source or {}
        if source_text_mode == "snippet":
            return str(source.get("snippet", "") or "").strip()
        return _source_content(source.get("url", ""))

    def _load_bert_nli_model():
        try:
            tokenizer = AutoTokenizer.from_pretrained(BERT_NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(BERT_NLI_MODEL_NAME)
            model.eval()

            return {
                "torch": torch,
                "tokenizer": tokenizer,
                "model": model,
            }
        except Exception as e:
            logger.warning("Could not initialize BERT NLI model %s: %s", BERT_NLI_MODEL_NAME, e)
            return None

    bert_nli_model = _load_bert_nli_model()

    def _bert_nli_scores(source_text, sentence):
        if bert_nli_model is None:
            return {"label": "", "confidence": 0.0, "reasoning": ""}

        source_text = str(source_text or "").strip()
        sentence = str(sentence or "").strip()
        if not source_text or not sentence:
            return {"label": "", "confidence": 0.0, "reasoning": ""}

        torch = bert_nli_model["torch"]
        tokenizer = bert_nli_model["tokenizer"]
        model = bert_nli_model["model"]

        try:
            encoded = tokenizer(
                source_text,
                sentence,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )
            with torch.no_grad():
                logits = model(**encoded).logits[0]
                # label_mapping = ['entailment', 'neutral', 'contradiction']
                label_mapping = ['contradiction', 'neutral', 'entailment']
                probs = torch.softmax(logits, dim=-1)
                # print(probs)
                label_id = int(torch.argmax(probs).item())
                confidence = float(probs[label_id].item())

            label = label_mapping[label_id]
            # print(label)
            payload = {
                "label": label,
                "confidence": confidence,
                "reasoning": "",
            }
        except Exception as e:
            logger.warning("BERT NLI scoring failed: %s", e)
            payload = {"label": "", "confidence": 0.0, "reasoning": ""}

        return payload

    def _nli_label(payload):
        payload = payload if isinstance(payload, dict) else {}
        return str(payload.get("label", "")).strip().lower()

    def _nli_score(payload):
        payload = payload if isinstance(payload, dict) else {}
        try:
            return float(payload.get("confidence", payload.get("score", 0)) or 0)
        except (TypeError, ValueError):
            return 0

    def _nli_reasoning(payload):
        payload = payload if isinstance(payload, dict) else {}
        return payload.get("reasoning", payload.get("reason", ""))

    def _score_candidate(source, sentence, source_relation, source_group):
        source_text = _source_text(source)
        if nli_method == "judge":
            if source_text.strip() and sentence.strip():
                nli_judge = compute_nli_scores(source_text, sentence)
            else:
                nli_judge = {"label": "", "confidence": 0.0, "reasoning": ""}
            bert_nli = {"label": "", "confidence": 0.0, "reasoning": ""}
        else:
            nli_judge = {"label": "", "confidence": 0.0, "reasoning": ""}
            bert_nli = _bert_nli_scores(source_text, sentence)

        judge_label = _nli_label(nli_judge)
        judge_score = _nli_score(nli_judge)
        judge_entailed = (
            judge_label == "entailment"
            and judge_score >= judge_entailment_min_score
        )
        bert_label = _nli_label(bert_nli)
        bert_confidence = _nli_score(bert_nli)
        bert_entailed = bert_label == "entailment"
        attribution_entailed = judge_entailed if nli_method == "judge" else bert_entailed
        source_bucket = {
            "cited_marker": "Marked Citations",
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
            "source_text_mode": source_text_mode,
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

    def _candidate_sources(source_records, citation_refs):
        candidates = []
        seen_urls = set()
        citation_refs = set(citation_refs or [])
        cited_sources = [
            source
            for source in source_records
            if source["source_type"] == "Cited"
        ]
        retrieved_sources = [
            source
            for source in source_records
            if source["source_type"] == "Retrieved"
        ]
        cited_by_url = {
            source["url"]: source
            for source in cited_sources
            if source.get("url")
        }
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
            source
            for source in cited_sources
            if source["ref_key"] in citation_refs
        ]
        marked_retrieved_sources = [
            source
            for source in retrieved_sources
            if source["ref_key"] in citation_refs
        ]
        marker_sources.extend(
            cited_by_url[source["url"]]
            for source in marked_retrieved_sources
            if source["url"] in cited_by_url
        )
        _append_sources(marker_sources, "cited_marker", "Cited Sources")

        other_cited_sources = [
            source
            for source in cited_sources
            if source["url"] not in seen_urls
        ]
        _append_sources(other_cited_sources, "other_cited", "Cited Sources")

        retrieved_sources = [
            source
            for source in retrieved_sources
            if source["url"] not in cited_urls
            and source["url"] not in seen_urls
        ]
        _append_sources(retrieved_sources, "retrieved", "Retrieved Sources")

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
        if isinstance(value, pd.Timestamp):
            return str(value)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    def _base_output_row(row, sample_index, sentence_index, response_text, sentence_payload):
        sentence = sentence_payload["sentence"]
        return {
            "sample": sample_index,
            "nli_method": nli_method,
            "chunking_method": chunking_method,
            "claim_selection_mode": (
                claim_selection_mode if chunking_method == "claim" else ""
            ),
            "source_text_mode": source_text_mode,
            "user_id": row.get("user_id"),
            "conv_id": row.get("conv_id"),
            "turn_id": row.get("turn_id"),
            "user_query": row.get("user_query", ""),
            "topic": row.get("topic"),
            "language": row.get("language"),
            "time": _json_safe(row.get("time")),
            "response_text": response_text,
            "response_chunk_index": sentence_index,
            "response_chunk_text": sentence,
            "citation_refs": sentence_payload["citation_refs"],
            "citation_markers": sentence_payload["citation_markers"],
        }

    def _judge_checked_source(check):
        return {
            "url": check["url"],
            "domain": check["domain"],
            "title": check["title"],
            "source_type": check["source_type"],
            "source_bucket": check["source_bucket"],
            "source_text_mode": check["source_text_mode"],
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

    rows = []
    for sample_index, row in tqdm(df.iterrows(), total=len(df)):
        response_text = str(row.get("asistant_response", "") or "")
        response_sentences = _extract_response_sentences(response_text)
        sources = _source_records(row)

        for sentence_index, sentence_payload in enumerate(response_sentences):
            sentence = sentence_payload["sentence"]
            citation_refs = sentence_payload["citation_refs"]
            candidates = _candidate_sources(sources, citation_refs)
            checked_sources = []
            entailed_check = None
            base_row = _base_output_row(
                row,
                sample_index,
                sentence_index,
                response_text,
                sentence_payload,
            )

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

            marker_checks = _evaluate_candidates(marker_candidates)
            checked_sources.extend(marker_checks)
            entailed_check = _first_entailed(marker_checks)

            if entailed_check is None:
                other_cited_checks = _evaluate_candidates(
                    other_cited_candidates,
                    stop_on_entailment=True,
                )
                checked_sources.extend(other_cited_checks)
                entailed_check = _first_entailed(other_cited_checks)

            if entailed_check is None:
                retrieved_checks = _evaluate_candidates(
                    retrieved_candidates,
                    stop_on_entailment=True,
                )
                checked_sources.extend(retrieved_checks)
                entailed_check = _first_entailed(retrieved_checks)

            marker_cited_urls = [
                candidate["url"]
                for candidate in marker_candidates
            ]

            if entailed_check is None:
                entailed_check = {
                    "nli_method": nli_method,
                    "url": "",
                    "domain": "",
                    "title": "",
                    "source_type": "Unknown",
                    "source_relation": "Unknown",
                    "source_group": "Unknown",
                    "source_bucket": "Unexplained",
                    "source_text_mode": source_text_mode,
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

            rows.append(
                {
                    **base_row,
                    "marker_cited_urls": marker_cited_urls,
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
                    "Unknown": (
                        entailed_check["source_group"] == "Unknown"
                    ),
                    "checked_sources": [
                        _judge_checked_source(check)
                        for check in checked_sources
                    ],
                }
            )

    if claims_cache_dirty:
        _save_claims_cache(persisted_claims_cache, cache_path=claim_cache_path)

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    result_df.to_csv(f"{output_base}.csv", index=False)
    result_df.to_pickle(f"{output_base}.pkl")

    json_records = [
        {key: _json_safe(value) for key, value in record.items()}
        for record in result_df.to_dict(orient="records")
    ]
    to_json(json_records, f"{output_base}.json")

    return result_df


def response_source_nli_sentence_based_factuality(
    input_path=None,
    output_base=None,
    model_name=FACTUALITY_JUDGE_MODEL,
    output_suffix="_factuality",
    checkpoint_every=10,
    platform="chatgpt",
):
    """
    Load sentence/claim-level NLI output, judge each extracted claim's factuality,
    and save the enriched records to a sibling factuality file.
    """
    records, resolved_input_path = _load_response_source_nli_sentence_based_records(
        input_path=input_path,
        output_base=output_base,
    )

    if input_path:
        base_without_ext = os.path.splitext(resolved_input_path)[0]
    else:
        base_without_ext = output_base or os.path.splitext(resolved_input_path)[0]

    factuality_output_base = f"{base_without_ext}{output_suffix}"
    factuality_output_dir = os.path.dirname(factuality_output_base)
    if factuality_output_dir:
        os.makedirs(factuality_output_dir, exist_ok=True)
    user_query_lookup = _build_response_source_user_query_lookup(platform=platform)
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
        to_json(json_records, f"{factuality_output_base}.json")
        return factuality_df

    for record_index, record in enumerate(tqdm(records, total=len(records)), start=1):
        claim = str(record.get("response_chunk_text", "") or "").strip()
        join_key = _record_join_key(record)
        user_query = str(
            record.get("user_query", "") or user_query_lookup.get(join_key, "")
        ).strip()
        cache_key = (user_query, claim)
        if cache_key in claim_cache:
            factuality = claim_cache[cache_key]
        else:
            factuality = evaluate_claim_factuality(
                claim,
                user_query=user_query,
                model_name=model_name,
            )
            claim_cache[cache_key] = factuality

        enriched_record = {
            "conv_id": str(record.get("conv_id", "") or "").strip(),
            "turn_id": str(record.get("turn_id", "") or "").strip(),
            "user_query": user_query,
            "claim": claim,
            "factuality_label": factuality.get("label", ""),
            "factuality_score": factuality.get("score", 0.0),
            "factuality_reasoning": factuality.get("reasoning", ""),
            "factuality_raw_response": factuality.get("raw_response", ""),
            "factuality_model": factuality.get("model", model_name),
        }
        enriched_records.append(enriched_record)
        if record_index % checkpoint_every == 0:
            _save_factuality_checkpoint(enriched_records)

    factuality_df = _save_factuality_checkpoint(enriched_records)
    return factuality_df


def response_source_claim_cache_factuality(
    cache_path=None,
    model_name=FACTUALITY_JUDGE_MODEL,
    output_suffix="_factuality",
    platform="chatgpt",
):
    """
    Load the cached extracted claims JSON, evaluate each claim's factuality,
    and save an enriched cache-shaped JSON file alongside the original cache.
    """
    if cache_path is None:
        cache_path = (
            CLAIM_EXTRACTION_CACHE_PATH
            if platform == "chatgpt"
            else f"{OUTPUT_PATH}/{platform}/metadata/response_source_claim_chunks_cache.json"
        )
    claims_cache = _load_claims_cache(cache_path=cache_path)
    if not claims_cache:
        raise FileNotFoundError(
            f"Claim cache not found or empty at {cache_path}."
        )

    cache_key_to_user_query = _build_claim_cache_user_query_lookup(platform=platform)
    claim_cache = {}
    enriched_cache = {}

    for cache_key, claim_entries in tqdm(claims_cache.items(), total=len(claims_cache)):
        fallback_user_query = str(cache_key_to_user_query.get(str(cache_key), "") or "").strip()
        enriched_claims = []
        for claim_entry in claim_entries:
            if isinstance(claim_entry, dict):
                claim_text = str(claim_entry.get("claim", "") or "").strip()
                user_query = str(
                    claim_entry.get("user_query", "") or fallback_user_query
                ).strip()
            else:
                claim_text = str(claim_entry or "").strip()
                user_query = fallback_user_query
            if not claim_text:
                continue
            factuality_cache_key = (user_query, claim_text)
            if factuality_cache_key in claim_cache:
                factuality = claim_cache[factuality_cache_key]
            else:
                factuality = evaluate_claim_factuality(
                    claim_text,
                    user_query=user_query,
                    model_name=model_name,
                )
                claim_cache[factuality_cache_key] = factuality

            enriched_claims.append(
                {
                    "claim": claim_text,
                    "user_query": user_query,
                    "factuality_label": factuality.get("label", ""),
                    "factuality_score": factuality.get("score", 0.0),
                    "factuality_reasoning": factuality.get("reasoning", ""),
                    "factuality_raw_response": factuality.get("raw_response", ""),
                    "factuality_model": factuality.get("model", model_name),
                }
            )
        enriched_cache[str(cache_key)] = enriched_claims

    output_path = _claim_cache_factuality_output_path(
        cache_path=cache_path,
        output_suffix=output_suffix,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    to_json(_json_safe(enriched_cache), output_path, indent=2)

    return enriched_cache


def summarize_response_source_nli_sentence_based_factuality(
    input_path=f"{OUTPUT_PATH}/chatgpt/metadata/response_source_nli_sentence_based_judge_claim.json",
    factuality_input_path=None,
    n_boot=1000,
    random_state=42,
    grounding_level="conversation",
):
    """
    Load claim-level NLI output and paired factuality checkpoints for ChatGPT and
    other platforms, then summarize factuality by four attribution groups:
    Associated Citations, Other Citations, Retrieved-not-cited, and Parametric
    Knowledge. Adds bootstrap 95% confidence intervals for mean factuality by
    resampling responses with replacement. For cited claims, the split into
    Cited Retrieved vs Cited Parametric follows the requested grounding level.
    """
    from src.response_generation.source_selection import (
        _normalize_url_for_source_matching,
        _row_retrieved_urls_by_grounding,
        _iter_rows_for_grounding,
        _validated_grounding_level,
    )

    grounding_level = _validated_grounding_level(grounding_level)
    source_group_order = [
        "Associated Citations",
        "Other Citations",
        "Retrieved-not-cited",
        "Parametric Knowledge",
    ]
    platform_path_map = {
        "ChatGPT": (
            input_path,
            factuality_input_path
            or f"{os.path.splitext(input_path)[0]}_factuality.json",
        ),
        "Claude": (
            f"{OUTPUT_PATH}/claude/metadata/response_source_nli_sentence_based_judge_claim.json",
            f"{OUTPUT_PATH}/claude/metadata/response_source_nli_sentence_based_judge_claim_factuality.json",
        ),
        "Grok": (
            f"{OUTPUT_PATH}/grok/metadata/response_source_nli_sentence_based_judge_claim.json",
            f"{OUTPUT_PATH}/grok/metadata/response_source_nli_sentence_based_judge_claim_factuality.json",
        ),
        "DeepSeek": (
            f"{OUTPUT_PATH}/deepseek/metadata/response_source_nli_sentence_based_judge_claim.json",
            f"{OUTPUT_PATH}/deepseek/metadata/response_source_nli_sentence_based_judge_claim_factuality.json",
        ),
    }

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
        return ""

    def _load_retrieved_url_lookup_for_response_source_path(response_source_path):
        metadata_dir = os.path.dirname(response_source_path)
        pkl_path = os.path.join(metadata_dir, "response_and_sources.pkl")
        csv_path = os.path.join(metadata_dir, "response_and_sources.csv")

        if os.path.exists(pkl_path):
            response_df = pd.read_pickle(pkl_path).copy()
        elif os.path.exists(csv_path):
            response_df = pd.read_csv(csv_path).copy()
            for source_col in ["srcs_retrieved"]:
                if source_col in response_df.columns:
                    response_df[source_col] = response_df[source_col].apply(_safe_parse_source_list)
        else:
            return {}

        retrieved_url_lookup = {}
        previous_retrieved_by_conv = {}
        for row in _iter_rows_for_grounding(response_df, grounding_level):
            join_key = (
                str(getattr(row, "user_id", "") or "").strip(),
                str(getattr(row, "conv_id", "") or "").strip(),
                str(getattr(row, "turn_id", "") or "").strip(),
            )
            if not any(join_key):
                continue
            retrieved_urls = {
                _normalize_url_for_source_matching(url)
                for url in _row_retrieved_urls_by_grounding(
                    row,
                    previous_retrieved_by_conv,
                    grounding_level,
                )
                if url
            }
            retrieved_url_lookup[join_key] = retrieved_urls
        return retrieved_url_lookup

    def _bootstrap_group_mean_cis(group_df, group_col, group_order):
        if len(group_df) == 0:
            return {}

        work_df = group_df.copy()
        response_key_cols = ["user_id", "conv_id", "turn_id"]
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
                    boot_df.loc[
                        boot_df[group_col] == group_name, "factuality_score"
                    ],
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

    def _build_platform_rows(platform_label, source_input_path, factuality_input_path_local):
        if not os.path.exists(source_input_path) or not os.path.exists(factuality_input_path_local):
            logger.warning(
                "Skipping %s factuality summary because input files are missing: %s ; %s",
                platform_label,
                source_input_path,
                factuality_input_path_local,
            )
            return pd.DataFrame()

        source_records = load_json(source_input_path)
        factuality_records = load_json(factuality_input_path_local)
        if not isinstance(source_records, list):
            raise ValueError(f"Expected a JSON list at {source_input_path}")
        if not isinstance(factuality_records, list):
            raise ValueError(f"Expected a JSON list at {factuality_input_path_local}")

        factuality_lookup = {}
        for record in factuality_records:
            if not isinstance(record, dict):
                continue
            key = (
                str(record.get("conv_id", "") or "").strip(),
                str(record.get("turn_id", "") or "").strip(),
                str(record.get("claim", "") or "").strip(),
            )
            if key[2]:
                factuality_lookup[key] = record

        retrieved_url_lookup = _load_retrieved_url_lookup_for_response_source_path(
            source_input_path
        )
        rows = []
        for record in source_records:
            if not isinstance(record, dict):
                continue
            claim = str(record.get("response_chunk_text", "") or "").strip()
            if not claim:
                continue
            source_group = _normalize_source_group(record.get("entailment_source_bucket", ""))
            if source_group not in source_group_order:
                continue
            join_key = (
                str(record.get("conv_id", "") or "").strip(),
                str(record.get("turn_id", "") or "").strip(),
                claim,
            )
            source_lookup_key = (
                str(record.get("user_id", "") or "").strip(),
                join_key[0],
                join_key[1],
            )
            factuality_record = factuality_lookup.get(join_key, {})
            score = pd.to_numeric(
                factuality_record.get("factuality_score", np.nan),
                errors="coerce",
            )
            entailed_url = _normalize_url_for_source_matching(
                record.get("entailed_url", "")
            )
            retrieved_urls = retrieved_url_lookup.get(source_lookup_key, set())
            citation_retrieval_bucket = ""
            if source_group in {"Associated Citations", "Other Citations"}:
                citation_retrieval_bucket = (
                    "Cited Retrieved"
                    if entailed_url and entailed_url in retrieved_urls
                    else "Cited Parametric"
                )
            rows.append(
                {
                    "platform": platform_label,
                    "user_id": str(record.get("user_id", "") or "").strip(),
                    "conv_id": join_key[0],
                    "turn_id": join_key[1],
                    "claim": claim,
                    "source_group": source_group,
                    "citation_retrieval_bucket": citation_retrieval_bucket,
                    "factuality_score": score,
                }
            )
        return pd.DataFrame(rows)

    platform_frames = []
    for platform_label in EXTERNAL_PLATFORM_ORDER:
        source_input_path, factuality_input_path_local = platform_path_map.get(platform_label, (None, None))
        if not source_input_path or not factuality_input_path_local:
            continue
        frame = _build_platform_rows(
            platform_label,
            source_input_path,
            factuality_input_path_local,
        )
        if len(frame) > 0:
            platform_frames.append(frame)

    if not platform_frames:
        raise ValueError("No joined records found between source and factuality files.")

    all_rows_df = pd.concat(platform_frames, ignore_index=True, sort=False)

    def _summarize(group_df, platform_label, group_col, group_order, summary_level):
        if len(group_df) == 0:
            return pd.DataFrame()
        ci_lookup = _bootstrap_group_mean_cis(group_df, group_col, group_order)
        total_chunks = float(len(group_df))
        summary = (
            group_df.groupby(group_col, dropna=False)
            .agg(
                chunk_count=("claim", "size"),
                avg_factuality_score=("factuality_score", "mean"),
            )
            .reindex(group_order, fill_value=0)
            .reset_index()
        )
        summary["chunk_share"] = (
            summary["chunk_count"].astype(float) / total_chunks
            if total_chunks > 0
            else 0.0
        )
        summary["ci_low"] = summary[group_col].map(
            lambda key: ci_lookup.get(key, (np.nan, np.nan))[0]
        )
        summary["ci_high"] = summary[group_col].map(
            lambda key: ci_lookup.get(key, (np.nan, np.nan))[1]
        )
        summary["avg_factuality_score"] = pd.to_numeric(
            summary["avg_factuality_score"], errors="coerce"
        ).round(4)
        summary["chunk_share"] = pd.to_numeric(
            summary["chunk_share"], errors="coerce"
        ).round(4)
        summary["ci_low"] = pd.to_numeric(summary["ci_low"], errors="coerce").round(4)
        summary["ci_high"] = pd.to_numeric(summary["ci_high"], errors="coerce").round(4)
        summary["platform"] = platform_label
        summary["n_boot"] = int(max(1, int(n_boot or 1)))
        summary["n_responses"] = int(
            len(group_df[["user_id", "conv_id", "turn_id"]].drop_duplicates())
        )
        summary["summary_level"] = summary_level
        return summary

    summary_frames = []
    citation_retrieval_order = ["Cited Retrieved", "Cited Parametric"]
    summary_frames.append(
        _summarize(
            all_rows_df,
            "All",
            "source_group",
            source_group_order,
            "source_group",
        )
    )
    all_cited_df = all_rows_df[
        all_rows_df["citation_retrieval_bucket"].isin(citation_retrieval_order)
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
    for platform_label in EXTERNAL_PLATFORM_ORDER:
        platform_df = all_rows_df[all_rows_df["platform"] == platform_label].copy()
        if len(platform_df) == 0:
            continue
        summary_frames.append(
            _summarize(
                platform_df,
                platform_label,
                "source_group",
                source_group_order,
                "source_group",
            )
        )
        platform_cited_df = platform_df[
            platform_df["citation_retrieval_bucket"].isin(citation_retrieval_order)
        ].copy()
        if len(platform_cited_df) > 0:
            summary_frames.append(
                _summarize(
                    platform_cited_df,
                    platform_label,
                    "citation_retrieval_bucket",
                    citation_retrieval_order,
                    "citation_retrieval_bucket",
                )
            )

    combined_summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    combined_summary["platform"] = pd.Categorical(
        combined_summary["platform"],
        categories=["All"] + EXTERNAL_PLATFORM_ORDER,
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
        ["platform", "summary_level", "source_group", "citation_retrieval_bucket"],
        kind="stable",
    ).reset_index(drop=True)

    print(combined_summary.to_string(index=False))
    return combined_summary


def plot_claim_bucket_tranco_rank_comparison(
    input_path=None,
    tranco_input_path=None,
    file_name="claim_bucket_tranco_rank_comparison",
    platform="chatgpt",
):
    """
    Compare Tranco ranks for claim chunks attributed to Retrieved Sources versus
    claim chunks attributed to Marked/Other Citations using the full-content
    claim-level NLI output.
    """
    from src.response_generation.source_selection import (
        evaluate_source_tranco_ranks,
        _normalize_url_for_source_matching,
    )

    if input_path is None:
        input_path = f"{OUTPUT_PATH}/{platform}/metadata/response_source_nli_sentence_based_judge_claim.json"
    claim_records = load_json(input_path)
    if not isinstance(claim_records, list):
        raise ValueError(f"Expected a JSON list at {input_path}")

    tranco_input_path = tranco_input_path or (
        f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_with_tranco_ranks.pkl"
    )
    if not os.path.exists(tranco_input_path):
        raise FileNotFoundError(
            f"Tranco-ranked source metadata not found at {tranco_input_path}. "
            "Run the Tranco-ranking pipeline first."
        )

    # Reuse the existing Tranco evaluation helper so the standard rank artifacts
    # and sanity checks stay in sync with this analysis.
    try:
        evaluate_source_tranco_ranks(platform=platform)
    except Exception as e:
        logger.warning("evaluate_source_tranco_ranks() failed before plotting: %s", e)

    tranco_df = pd.read_pickle(tranco_input_path).copy()

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

    tranco_lookup = {}
    for _, row in tranco_df.iterrows():
        join_key = (
            str(row.get("user_id", "") or "").strip(),
            str(row.get("conv_id", "") or "").strip(),
            str(row.get("turn_id", "") or "").strip(),
        )
        if not any(join_key):
            continue

        bucket_maps = {}
        for source_col, rank_col in [
            ("srcs_retrieved", "ranks_srcs_retrieved"),
            ("srcs_cited", "ranks_srcs_cited"),
        ]:
            sources = _as_list(row.get(source_col, []))
            ranks = _as_list(row.get(rank_col, []))
            url_to_ranks = {}
            for idx, src in enumerate(sources):
                if not isinstance(src, dict):
                    continue
                normalized_url = _normalize_url_for_source_matching(src.get("url", ""))
                if not normalized_url:
                    continue
                try:
                    rank_value = float(ranks[idx])
                except (IndexError, TypeError, ValueError):
                    continue
                if rank_value <= 0:
                    continue
                url_to_ranks.setdefault(normalized_url, []).append(rank_value)
            bucket_maps[source_col] = url_to_ranks

        tranco_lookup[join_key] = bucket_maps

    bucket_label_map = {
        "Retrieved Sources": "Retrieved Sources",
        "Marked Citations": "Associated/Other Citations",
        "Other Citations": "Associated/Other Citations",
    }
    source_col_map = {
        "Retrieved Sources": "srcs_retrieved",
        "Marked Citations": "srcs_cited",
        "Other Citations": "srcs_cited",
    }

    rank_rows = []
    for record in claim_records:
        if not isinstance(record, dict):
            continue
        raw_bucket = str(record.get("entailment_source_bucket", "") or "").strip()
        plot_bucket = bucket_label_map.get(raw_bucket, "")
        if not plot_bucket:
            continue

        entailed_url = _normalize_url_for_source_matching(record.get("entailed_url", ""))
        if not entailed_url:
            continue

        join_key = (
            str(record.get("user_id", "") or "").strip(),
            str(record.get("conv_id", "") or "").strip(),
            str(record.get("turn_id", "") or "").strip(),
        )
        source_rank_maps = tranco_lookup.get(join_key, {})
        url_to_ranks = source_rank_maps.get(source_col_map[raw_bucket], {})
        rank_values = url_to_ranks.get(entailed_url, [])
        if not rank_values:
            continue

        # If the same URL appears multiple times in the aligned source list,
        # keep the mean of valid ranks for that URL within the response.
        tranco_rank = float(np.mean(rank_values))
        rank_rows.append(
            {
                "user_id": join_key[0],
                "conv_id": join_key[1],
                "turn_id": join_key[2],
                "claim": str(record.get("response_chunk_text", "") or "").strip(),
                "entailment_source_bucket": raw_bucket,
                "plot_bucket": plot_bucket,
                "entailed_url": entailed_url,
                "tranco_rank": tranco_rank,
                "log10_tranco_rank": float(np.log10(tranco_rank)),
            }
        )

    rank_df = pd.DataFrame(rank_rows)
    if len(rank_df) == 0:
        raise ValueError("No claim-level Tranco ranks found for the selected buckets.")

    output_dir = f"{OUTPUT_PATH}/{platform}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    rank_df.to_csv(f"{output_dir}/{file_name}.csv", index=False)

    summary_df = (
        rank_df.groupby("plot_bucket", dropna=False).agg(
            claim_count=("tranco_rank", "count"),
            avg_tranco_rank=("tranco_rank", "mean"),
            median_tranco_rank=("tranco_rank", "median"),
            avg_log10_tranco_rank=("log10_tranco_rank", "mean"),
            median_log10_tranco_rank=("log10_tranco_rank", "median"),
        )
        .rename(
            columns={}
        )
        .reset_index()
    )
    summary_df.to_csv(f"{output_dir}/{file_name}_summary.csv", index=False)

    bucket_order = ["Retrieved Sources", "Associated/Other Citations"]
    color_map = {
        "Retrieved Sources": "#636EFA",
        "Associated/Other Citations": "#EF553B",
    }

    all_log_values = []
    for bucket in bucket_order:
        subset = rank_df[rank_df["plot_bucket"] == bucket]["log10_tranco_rank"]
        subset = pd.to_numeric(subset, errors="coerce").to_numpy()
        subset = subset[np.isfinite(subset)]
        if len(subset) > 0:
            all_log_values.append(subset)

    if len(all_log_values) > 0:
        combined_values = np.concatenate(all_log_values)
        global_min_log = float(np.floor(np.min(combined_values)))
        global_max_log = float(np.ceil(np.max(combined_values)))
        if global_max_log <= global_min_log:
            global_max_log = global_min_log + 1.0
    else:
        global_min_log = 0.0
        global_max_log = 1.0

    fig = go.Figure()
    for bucket in bucket_order:
        subset = rank_df[rank_df["plot_bucket"] == bucket].copy()
        if len(subset) == 0:
            continue
        fig.add_trace(
            go.Violin(
                x=[bucket] * len(subset),
                y=subset["log10_tranco_rank"],
                name=bucket,
                legendgroup=bucket,
                marker_color=color_map[bucket],
                line_color=color_map[bucket],
                width=0.9,
                box_visible=True,
                meanline_visible=True,
                hovertemplate=(
                    "Bucket: %{x}<br>"
                    "log10(Tranco rank): %{y:.2f}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    fig.update_layout(
        xaxis_title="Claim Attribution Bucket",
        yaxis_title="log10(Tranco Rank)",
        xaxis=dict(tickangle=0),
        yaxis=dict(range=[global_min_log, global_max_log], tickmode="linear", dtick=1),
        violinmode="group",
        margin=dict(t=20, b=110, r=5),
    )

    fig.write_html(f"{output_dir}/{file_name}.html")
    try:
        paper_fig = with_paper_style(fig, config=styler(24, 16), legend_pos=None)
        paper_fig.update_xaxes(tickangle=0, tickfont=dict(size=24))
        paper_fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
    except Exception as e:
        logger.warning("Could not write claim-bucket Tranco rank PDF: %s", e)

    return rank_df


def diagnose_snippet_unexplained_rows(
    input_path=None,
    platform="chatgpt",
):
    """
    Diagnose why snippet-mode judge outputs produce many Unexplained rows.
    Prints bucket counts plus snippet-coverage stats for checked sources.
    """
    if input_path is None:
        input_path = f"{OUTPUT_PATH}/{platform}/metadata/response_source_nli_sentence_based_judge_claim_snippet.json"
    source_records = load_json(input_path)
    if not isinstance(source_records, list):
        raise ValueError(f"Expected a JSON list at {input_path}")

    response_df = _load_response_source_similarity_input(platform=platform).copy()
    snippet_lookup = {}
    for _, row in response_df.iterrows():
        record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        join_key = _record_join_key(record)
        url_to_snippet = {}
        for source_col in ["srcs_cited", "srcs_retrieved"]:
            for src in _safe_parse_source_list(record.get(source_col, [])):
                if not isinstance(src, dict):
                    continue
                url = str(src.get("url", "") or "").strip()
                if not url:
                    continue
                url_to_snippet.setdefault(url, str(src.get("snippet", "") or "").strip())
        snippet_lookup[join_key] = url_to_snippet

    total_rows = 0
    bucket_counts = Counter()
    unexplained_like_rows = 0
    rows_with_no_checked_sources = 0
    rows_with_any_empty_checked_snippet = 0
    rows_with_all_checked_snippets_empty = 0
    rows_with_some_nonempty_checked_snippet = 0

    unexplained_total = 0
    unexplained_no_checked = 0
    unexplained_all_checked_snippets_empty = 0
    unexplained_some_nonempty_checked_snippet = 0

    for record in source_records:
        if not isinstance(record, dict):
            continue
        total_rows += 1
        bucket = str(record.get("entailment_source_bucket", "") or "").strip()
        bucket_counts[bucket] += 1

        is_unexplained_like = bucket in {"Unexplained", "Unknown", ""}
        if is_unexplained_like:
            unexplained_like_rows += 1
            unexplained_total += 1

        checked_sources = record.get("checked_sources", []) or []
        if not checked_sources:
            rows_with_no_checked_sources += 1
            if is_unexplained_like:
                unexplained_no_checked += 1
            continue

        join_key = (
            str(record.get("user_id", "") or "").strip(),
            str(record.get("conv_id", "") or "").strip(),
            str(record.get("turn_id", "") or "").strip(),
        )
        url_to_snippet = snippet_lookup.get(join_key, {})
        checked_snippets = [
            url_to_snippet.get(str((src or {}).get("url", "") or "").strip(), "").strip()
            for src in checked_sources
        ]
        nonempty_count = sum(1 for snippet in checked_snippets if snippet)
        if any(not snippet for snippet in checked_snippets):
            rows_with_any_empty_checked_snippet += 1
        if nonempty_count == 0:
            rows_with_all_checked_snippets_empty += 1
            if is_unexplained_like:
                unexplained_all_checked_snippets_empty += 1
        else:
            rows_with_some_nonempty_checked_snippet += 1
            if is_unexplained_like:
                unexplained_some_nonempty_checked_snippet += 1

    def _rate(numerator, denominator):
        return round(float(numerator) / float(denominator), 4) if denominator else 0.0

    summary = {
        "total_rows": total_rows,
        "bucket_counts": dict(bucket_counts),
        "unexplained_like_rows": unexplained_like_rows,
        "unexplained_like_rate": _rate(unexplained_like_rows, total_rows),
        "rows_with_no_checked_sources": rows_with_no_checked_sources,
        "rows_with_any_empty_checked_snippet": rows_with_any_empty_checked_snippet,
        "rate_any_empty_checked_snippet": _rate(rows_with_any_empty_checked_snippet, total_rows),
        "rows_with_all_checked_snippets_empty": rows_with_all_checked_snippets_empty,
        "rate_all_checked_snippets_empty": _rate(rows_with_all_checked_snippets_empty, total_rows),
        "rows_with_some_nonempty_checked_snippet": rows_with_some_nonempty_checked_snippet,
        "rate_some_nonempty_checked_snippet": _rate(rows_with_some_nonempty_checked_snippet, total_rows),
        "unexplained_total": unexplained_total,
        "unexplained_no_checked": unexplained_no_checked,
        "unexplained_all_checked_snippets_empty": unexplained_all_checked_snippets_empty,
        "unexplained_some_nonempty_checked_snippet": unexplained_some_nonempty_checked_snippet,
        "unexplained_all_empty_rate": _rate(
            unexplained_all_checked_snippets_empty, unexplained_total
        ),
        "unexplained_some_nonempty_rate": _rate(
            unexplained_some_nonempty_checked_snippet, unexplained_total
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


def sample_response_source_nli_method_comparison(
    full_input_path=None,
    snippet_input_path=None,
    sample_size=10,
    random_state=42,
    output_path=None,
    platform="chatgpt",
):
    """
    Randomly sample aligned response chunks from full-text and snippet runs and
    save a side-by-side comparison with each method's selected URL, full
    content, snippet metadata, entailment-judge output, and final bucket.
    """
    if full_input_path is None:
        full_input_path = f"{OUTPUT_PATH}/{platform}/metadata/response_source_nli_sentence_based_judge_claim.json"
    if snippet_input_path is None:
        snippet_input_path = f"{OUTPUT_PATH}/{platform}/metadata/response_source_nli_sentence_based_judge_claim_snippet.json"
    full_records = load_json(full_input_path)
    snippet_records = load_json(snippet_input_path)
    if not isinstance(full_records, list):
        raise ValueError(f"Expected a JSON list at {full_input_path}")
    if not isinstance(snippet_records, list):
        raise ValueError(f"Expected a JSON list at {snippet_input_path}")

    if output_path is None:
        output_path = (
            f"{OUTPUT_PATH}/{platform}/metadata/"
            "response_source_nli_method_comparison_samples.json"
        )

    response_df = _load_response_source_similarity_input(platform=platform).copy()
    urls_content_path = (
        RESPONSE_URLS_CONTENT_PATH
        if platform == "chatgpt"
        else f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_url_content.json"
    )
    urls_content = _load_urls_content(
        urls_content_path=urls_content_path,
        required=False,
    )
    user_query_lookup = _build_response_source_user_query_lookup(platform=platform)

    def _clean_url(url):
        return str(url or "").strip().removesuffix("?utm_source=chatgpt.com").removesuffix(
            "&utm_source=chatgpt.com"
        )

    urls_content_by_clean_url = {}
    for url, content in urls_content.items():
        clean_url = _clean_url(url)
        if not clean_url:
            continue
        urls_content_by_clean_url[clean_url] = str(content or "")
        urls_content_by_clean_url[clean_url.rstrip("/")] = str(content or "")

    snippet_lookup = {}
    for _, row in response_df.iterrows():
        record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        join_key = _record_join_key(record)
        url_to_source_meta = {}
        for source_col in ["srcs_cited", "srcs_retrieved", "srcs_safe_urls"]:
            for src in _safe_parse_source_list(record.get(source_col, [])):
                if not isinstance(src, dict):
                    continue
                url = _clean_url(src.get("url", ""))
                if not url or url in url_to_source_meta:
                    continue
                url_to_source_meta[url] = {
                    "snippet": str(src.get("snippet", "") or "").strip(),
                    "title": str(src.get("title", "") or "").strip(),
                    "domain": str(src.get("domain", "") or "").strip(),
                }
        snippet_lookup[join_key] = url_to_source_meta

    join_cols = [
        "user_id",
        "conv_id",
        "turn_id",
        "response_chunk_index",
        "response_chunk_text",
    ]

    def _record_key(record):
        return tuple(str(record.get(col, "") or "").strip() for col in join_cols)

    full_by_key = {
        _record_key(record): record
        for record in full_records
        if isinstance(record, dict)
    }
    snippet_by_key = {
        _record_key(record): record
        for record in snippet_records
        if isinstance(record, dict)
    }

    shared_keys = [key for key in full_by_key.keys() if key in snippet_by_key]
    if not shared_keys:
        raise ValueError("No aligned rows found between full-text and snippet outputs.")

    sample_n = min(max(1, int(sample_size or 1)), len(shared_keys))
    sampled_keys = (
        pd.Series(shared_keys)
        .sample(n=sample_n, random_state=random_state, replace=False)
        .tolist()
    )

    def _source_payload(record):
        record = record if isinstance(record, dict) else {}
        entailed_url = _clean_url(record.get("entailed_url", ""))
        join_key = (
            str(record.get("user_id", "") or "").strip(),
            str(record.get("conv_id", "") or "").strip(),
            str(record.get("turn_id", "") or "").strip(),
        )
        source_meta = snippet_lookup.get(join_key, {}).get(entailed_url, {})
        full_content = urls_content_by_clean_url.get(
            entailed_url,
            urls_content_by_clean_url.get(entailed_url.rstrip("/"), ""),
        )
        return {
            "entailed_url": entailed_url,
            "entailed_title": str(record.get("entailed_title", "") or "").strip()
            or source_meta.get("title", ""),
            "entailed_domain": str(record.get("entailed_domain", "") or "").strip()
            or source_meta.get("domain", ""),
            "full_content": full_content,
            "snippet": source_meta.get("snippet", ""),
        }

    def _method_payload(record):
        record = record if isinstance(record, dict) else {}
        return {
            "source": _source_payload(record),
            "judge_nli_label": str(record.get("judge_nli_label", "") or "").strip(),
            "judge_nli_score": record.get("judge_nli_score", 0),
            "judge_nli_reasoning": str(
                record.get("judge_nli_reasoning", "") or ""
            ).strip(),
            "final_bucket": str(
                record.get("entailment_source_bucket", "") or ""
            ).strip(),
            "entailment_source_type": str(
                record.get("entailment_source_type", "") or ""
            ).strip(),
            "checked_source_count": int(record.get("checked_source_count", 0) or 0),
        }

    sampled_rows = []
    for key in sampled_keys:
        full_record = full_by_key[key]
        snippet_record = snippet_by_key[key]
        response_join_key = (
            str(full_record.get("user_id", "") or snippet_record.get("user_id", "") or "").strip(),
            str(full_record.get("conv_id", "") or snippet_record.get("conv_id", "") or "").strip(),
            str(full_record.get("turn_id", "") or snippet_record.get("turn_id", "") or "").strip(),
        )
        sampled_rows.append(
            {
                "user_id": key[0],
                "conv_id": key[1],
                "turn_id": key[2],
                "response_chunk_index": key[3],
                "response_chunk_text": key[4],
                "user_query": str(
                    full_record.get("user_query", "")
                    or snippet_record.get("user_query", "")
                    or user_query_lookup.get(response_join_key, "")
                    or ""
                ).strip(),
                "full_text_method": _method_payload(full_record),
                "snippet_method": _method_payload(snippet_record),
            }
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    to_json(_json_safe(sampled_rows), output_path, indent=2)
    print(json.dumps(_json_safe(sampled_rows), indent=2))
    return sampled_rows


def _normalize_entailment_source_bucket_series(source_bucket_series):
    return (
        source_bucket_series.fillna("Parametric Knowledge").replace(
            {
                "": "Parametric Knowledge",
                "Unexplained": "Parametric Knowledge",
                "Unknown": "Parametric Knowledge",
                "unknown": "Parametric Knowledge",
                "Marked Citations": "Associated Citations",
            }
        )
    )


def _bootstrap_bucket_rate_summary(
    sentence_df,
    bucket_order,
    response_key_cols,
    n_boot=1000,
    random_state=42,
):
    sentence_df = sentence_df.copy()
    if len(sentence_df) == 0:
        return pd.DataFrame()

    for col in response_key_cols:
        if col not in sentence_df.columns:
            sentence_df[col] = ""
        sentence_df[col] = sentence_df[col].astype(str)

    source_buckets = _normalize_entailment_source_bucket_series(
        sentence_df["entailment_source_bucket"]
    )
    sentence_df["normalized_source_bucket"] = source_buckets

    observed_counts = (
        sentence_df["normalized_source_bucket"]
        .value_counts()
        .reindex(bucket_order, fill_value=0)
    )
    total_sentences = float(len(sentence_df))
    observed_rates = (
        observed_counts.astype(float) / total_sentences
        if total_sentences > 0
        else pd.Series([0.0] * len(bucket_order), index=bucket_order)
    )

    response_keys = list(
        sentence_df[response_key_cols]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if not response_keys:
        return pd.DataFrame()

    response_groups = {
        key: group.copy()
        for key, group in sentence_df.groupby(response_key_cols, dropna=False, sort=False)
    }

    rng = np.random.default_rng(random_state)
    bootstrap_rates = {bucket: [] for bucket in bucket_order}

    for _ in range(max(1, int(n_boot or 1))):
        sampled_indices = rng.integers(0, len(response_keys), size=len(response_keys))
        sampled_frames = [response_groups[response_keys[idx]] for idx in sampled_indices]
        boot_df = pd.concat(sampled_frames, ignore_index=True)
        boot_counts = (
            boot_df["normalized_source_bucket"]
            .value_counts()
            .reindex(bucket_order, fill_value=0)
        )
        boot_total = float(len(boot_df))
        for bucket in bucket_order:
            rate = float(boot_counts[bucket]) / boot_total if boot_total > 0 else 0.0
            bootstrap_rates[bucket].append(rate)

    rows = []
    for bucket in bucket_order:
        values = np.asarray(bootstrap_rates[bucket], dtype=float)
        if len(values) == 0:
            ci_low = np.nan
            ci_high = np.nan
        else:
            ci_low = float(np.percentile(values, 2.5))
            ci_high = float(np.percentile(values, 97.5))
        rows.append(
            {
                "entailment_source_bucket": bucket,
                "sentence_count": int(observed_counts.get(bucket, 0)),
                "sentence_rate": float(observed_rates.get(bucket, 0.0)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_boot": int(max(1, int(n_boot or 1))),
                "n_responses": int(len(response_keys)),
                "n_claims": int(total_sentences),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_response_source_nli_sentence_based_confidence_intervals(
    output_base=None,
    output_csv_path=None,
    file_name=None,
    nli_method="bert",
    chunking_method="citation_marker",
    claim_selection_mode="latest_preceding",
    source_text_mode="full_url_content",
    n_boot=1000,
    random_state=42,
):
    """
    Bootstrap 95% confidence intervals for the four claim-attribution rates by
    resampling responses with replacement and recomputing claim-bucket shares.
    """
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    source_text_mode = _normalize_source_text_mode(source_text_mode)

    modes_to_summarize = (
        ["all", "latest_preceding"] if chunking_method == "claim" else [claim_selection_mode]
    )
    if file_name is None:
        file_name = f"response_source_nli_sentence_based_{nli_method}_bootstrap_ci"
        if chunking_method != "citation_marker":
            file_name = f"{file_name}_{chunking_method}"
        if source_text_mode != "full_url_content":
            file_name = f"{file_name}_{source_text_mode}"

    bucket_order = [
        "Associated Citations",
        "Other Citations",
        "Retrieved Sources",
        "Parametric Knowledge",
    ]
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)

    claim_all_df = None
    claim_latest_df = None
    external_claim_mode_dfs = {}
    if chunking_method == "claim":
        claim_all_output_base = (
            output_base
            if output_base is not None
            else _response_source_nli_output_base(
                nli_method,
                chunking_method,
                claim_selection_mode="all",
                source_text_mode=source_text_mode,
            )
        )
        claim_all_df = _load_response_source_nli_sentence_based(
            output_base=claim_all_output_base
        )

        def _citation_refs_key(value):
            if isinstance(value, (list, tuple)):
                parsed = value
            elif isinstance(value, str):
                text = value.strip()
                if not text:
                    return tuple()
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return tuple()
            else:
                return tuple()
            if not isinstance(parsed, (list, tuple)):
                return tuple()
            refs = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    refs.append(f"{str(item[0])}::{str(item[1])}")
                elif item is not None:
                    refs.append(str(item))
            return tuple(refs)

        def _prepare_claim_mode_dfs(df):
            df = df.copy()
            if "chunking_method" in df.columns:
                df = df[
                    df["chunking_method"].fillna("citation_marker") == "claim"
                ].copy()
            if len(df) == 0:
                return df, df
            sort_cols = [col for col in ["sample", "response_chunk_index"] if col in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, kind="stable").copy()
            if "citation_refs" not in df.columns:
                return df, df.iloc[0:0].copy()
            refs_key_series = df["citation_refs"].apply(_citation_refs_key)
            nonempty_refs_mask = refs_key_series.apply(bool)
            if not bool(nonempty_refs_mask.any()):
                return df, df.iloc[0:0].copy()
            sample_series = (
                df["sample"]
                if "sample" in df.columns
                else pd.Series([0] * len(df), index=df.index)
            )
            prev_sample = sample_series.shift(1)
            prev_refs = refs_key_series.shift(1)
            same_as_prev = (
                nonempty_refs_mask
                & (sample_series == prev_sample)
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
            return df, latest_df

        claim_all_df, claim_latest_df = _prepare_claim_mode_dfs(claim_all_df)

        for platform_label in EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys():
            platform_output_base = (
                EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES
                .get(platform_label, {})
                .get(nli_method)
            )
            if not platform_output_base:
                continue
            candidate_output_bases = _external_platform_output_base_candidates(
                platform_output_base,
                source_text_mode=source_text_mode,
            )
            platform_raw_df = None
            for candidate_output_base in candidate_output_bases:
                try:
                    platform_raw_df = _load_response_source_nli_sentence_based(
                        output_base=candidate_output_base
                    )
                    break
                except FileNotFoundError:
                    continue
            if platform_raw_df is None:
                continue
            platform_all_df, platform_latest_df = _prepare_claim_mode_dfs(platform_raw_df)
            if len(platform_all_df) == 0:
                continue
            external_claim_mode_dfs[platform_label] = {
                "all": platform_all_df,
                "latest_preceding": platform_latest_df,
            }

    summary_frames = []
    for mode in modes_to_summarize:
        platform_sentence_dfs = {}
        if chunking_method == "claim":
            platform_sentence_dfs["ChatGPT"] = (
                claim_all_df.copy() if mode == "all" else claim_latest_df.copy()
            )
            for platform_label in EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys():
                mode_dfs = external_claim_mode_dfs.get(platform_label)
                if not mode_dfs:
                    continue
                platform_df = mode_dfs.get(mode)
                if platform_df is None or len(platform_df) == 0:
                    continue
                platform_sentence_dfs[platform_label] = platform_df.copy()
        else:
            mode_output_base = (
                output_base
                if output_base is not None and len(modes_to_summarize) == 1
                else _response_source_nli_output_base(
                    nli_method,
                    chunking_method,
                    claim_selection_mode=mode,
                    source_text_mode=source_text_mode,
                )
            )
            sentence_df = _load_response_source_nli_sentence_based(
                output_base=mode_output_base
            )
            if "chunking_method" in sentence_df.columns:
                sentence_df = sentence_df[
                    sentence_df["chunking_method"].fillna("citation_marker")
                    == chunking_method
                ].copy()
            platform_sentence_dfs["ChatGPT"] = sentence_df

        for platform_label, sentence_df in platform_sentence_dfs.items():
            if len(sentence_df) == 0:
                continue
            response_key_cols = [
                col
                for col in ["sample", "user_id", "conv_id", "turn_id"]
                if col in sentence_df.columns
            ]
            if not response_key_cols:
                raise ValueError("No response-level key columns available for bootstrap.")
            summary_df = _bootstrap_bucket_rate_summary(
                sentence_df,
                bucket_order=bucket_order,
                response_key_cols=response_key_cols,
                n_boot=n_boot,
                random_state=random_state,
            )
            if len(summary_df) == 0:
                continue
            summary_df["platform"] = platform_label
            summary_df["claim_selection_mode"] = mode
            summary_df["nli_method"] = nli_method
            summary_df["chunking_method"] = chunking_method
            summary_df["source_text_mode"] = source_text_mode
            summary_frames.append(summary_df)

    if not summary_frames:
        return pd.DataFrame()

    combined_summary_df = pd.concat(summary_frames, ignore_index=True)
    output_csv_path = output_csv_path or f"{output_dir}/{file_name}.csv"
    combined_summary_df.to_csv(output_csv_path, index=False)
    to_json(
        [
            {key: _json_safe(value) for key, value in record.items()}
            for record in combined_summary_df.to_dict(orient="records")
        ],
        f"{output_dir}/{file_name}.json",
    )
    return combined_summary_df


def plot_response_source_nli_sentence_based(
    output_base=None,
    file_name=None,
    nli_method="bert",
    chunking_method="citation_marker",
    claim_selection_mode="latest_preceding",
    source_text_mode="full_url_content",
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    source_text_mode = _normalize_source_text_mode(source_text_mode)

    modes_to_plot = (
        ["all", "latest_preceding"] if chunking_method == "claim" else [claim_selection_mode]
    )

    if file_name is None:
        file_name = f"response_source_nli_sentence_based_{nli_method}_summary"
        if chunking_method != "citation_marker":
            file_name = f"{file_name}_{chunking_method}"
        if source_text_mode != "full_url_content":
            file_name = f"{file_name}_{source_text_mode}"

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
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    summary_frames = []

    claim_all_df = None
    claim_latest_df = None
    external_claim_mode_dfs = {}
    if chunking_method == "claim":
        claim_all_output_base = (
            output_base
            if output_base is not None
            else _response_source_nli_output_base(
                nli_method,
                chunking_method,
                claim_selection_mode="all",
                source_text_mode=source_text_mode,
            )
        )
        try:
            claim_all_df = _load_response_source_nli_sentence_based(
                output_base=claim_all_output_base
            )
        except FileNotFoundError as e:
            logger.warning("Could not load claim-based metadata for plotting: %s", e)
            return pd.DataFrame()

        def _citation_refs_key(value):
            if isinstance(value, (list, tuple)):
                parsed = value
            elif isinstance(value, str):
                text = value.strip()
                if not text:
                    return tuple()
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return tuple()
            else:
                return tuple()

            if not isinstance(parsed, (list, tuple)):
                return tuple()

            refs = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    refs.append(f"{str(item[0])}::{str(item[1])}")
                elif item is not None:
                    refs.append(str(item))
            return tuple(refs)

        def _prepare_claim_mode_dfs(df):
            df = df.copy()
            if "chunking_method" in df.columns:
                df = df[
                    df["chunking_method"].fillna("citation_marker") == "claim"
                ].copy()
            if len(df) == 0:
                return df, df

            sort_cols = [
                col
                for col in ["sample", "response_chunk_index"]
                if col in df.columns
            ]
            if sort_cols:
                df = df.sort_values(sort_cols, kind="stable").copy()

            if "citation_refs" not in df.columns:
                return df, df.iloc[0:0].copy()

            refs_key_series = df["citation_refs"].apply(_citation_refs_key)
            nonempty_refs_mask = refs_key_series.apply(bool)
            if not bool(nonempty_refs_mask.any()):
                return df, df.iloc[0:0].copy()

            sample_series = (
                df["sample"]
                if "sample" in df.columns
                else pd.Series([0] * len(df), index=df.index)
            )
            prev_sample = sample_series.shift(1)
            prev_refs = refs_key_series.shift(1)
            same_as_prev = (
                nonempty_refs_mask
                & (sample_series == prev_sample)
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

            return df, latest_df

        claim_all_df, claim_latest_df = _prepare_claim_mode_dfs(claim_all_df)
        if len(claim_all_df) == 0:
            return pd.DataFrame()

        # External files are treated as all-claims sources; derive latest mode here.
        for platform_label in EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys():
            platform_output_base = (
                EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES
                .get(platform_label, {})
                .get(nli_method)
            )
            if not platform_output_base:
                continue

            candidate_output_bases = _external_platform_output_base_candidates(
                platform_output_base,
                source_text_mode=source_text_mode,
            )

            platform_raw_df = None
            last_error = None
            for candidate_output_base in candidate_output_bases:
                try:
                    platform_raw_df = _load_response_source_nli_sentence_based(
                        output_base=candidate_output_base
                    )
                    break
                except FileNotFoundError as e:
                    last_error = e
                    continue

            if platform_raw_df is None:
                logger.warning(
                    "Skipping %s claim plot data: %s",
                    platform_label,
                    last_error or "no matching metadata file found",
                )
                continue
            platform_all_df, platform_latest_df = _prepare_claim_mode_dfs(platform_raw_df)
            if len(platform_all_df) == 0:
                continue
            external_claim_mode_dfs[platform_label] = {
                "all": platform_all_df,
                "latest_preceding": platform_latest_df,
            }

    for mode in modes_to_plot:
        platform_sentence_dfs = {}
        if chunking_method == "claim":
            platform_sentence_dfs["ChatGPT"] = (
                claim_all_df.copy()
                if mode == "all"
                else claim_latest_df.copy()
            )

            for platform_label in EXTERNAL_PLATFORM_CLAIM_LATEST_PRECEDING_BASES.keys():
                mode_dfs = external_claim_mode_dfs.get(platform_label)
                if not mode_dfs:
                    continue
                platform_df = mode_dfs.get(mode)
                if platform_df is None or len(platform_df) == 0:
                    continue
                platform_sentence_dfs[platform_label] = platform_df.copy()
        else:
            mode_output_base = (
                output_base
                if output_base is not None and len(modes_to_plot) == 1
                else _response_source_nli_output_base(
                    nli_method,
                    chunking_method,
                    claim_selection_mode=mode,
                    source_text_mode=source_text_mode,
                )
            )
            try:
                sentence_df = _load_response_source_nli_sentence_based(
                    output_base=mode_output_base
                )
            except FileNotFoundError as e:
                logger.warning(
                    "Skipping sentence-based NLI summary for claim_selection_mode=%s: %s",
                    mode,
                    e,
                )
                continue

            if "chunking_method" in sentence_df.columns:
                sentence_df = sentence_df[
                    sentence_df["chunking_method"].fillna("citation_marker")
                    == chunking_method
                ].copy()
            platform_sentence_dfs["ChatGPT"] = sentence_df

        mode_summary_frames = []
        for platform_label, sentence_df in platform_sentence_dfs.items():
            if len(sentence_df) == 0:
                continue

            source_buckets = (
                sentence_df["entailment_source_bucket"]
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
            sentence_weights = pd.Series([1] * len(sentence_df), index=sentence_df.index)
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
            platform_summary_df = pd.DataFrame(
                {
                    "entailment_source_bucket": source_order,
                    "sentence_count": [
                        int(counts.get(source_bucket, 0))
                        for source_bucket in source_order
                    ],
                }
            )
            platform_summary_df["sentence_rate"] = (
                platform_summary_df["sentence_count"] / total_sentences
            )
            platform_summary_df["total_sentence_count"] = int(total_sentences)
            platform_summary_df["claim_selection_mode"] = mode
            platform_summary_df["platform"] = platform_label
            mode_summary_frames.append(platform_summary_df)

        if not mode_summary_frames:
            continue
        mode_summary_df = pd.concat(mode_summary_frames, ignore_index=True)

        mode_file_name = (
            f"{file_name}_{mode}" if chunking_method == "claim" else file_name
        )
        mode_title = (
            mode_label_map.get(mode, mode) if chunking_method == "claim" else ""
        )
        mode_summary_df.to_csv(
            f"{output_dir}/{mode_file_name}.csv",
            index=False,
        )

        platform_order = EXTERNAL_PLATFORM_ORDER.copy()
        present_platforms = mode_summary_df["platform"].astype(str).unique().tolist()
        platform_order = [
            platform
            for platform in platform_order
            if platform in present_platforms
        ] + [
            platform
            for platform in present_platforms
            if platform not in platform_order
        ]
        platform_total_by_label = {}
        for platform_label in platform_order:
            platform_rows = mode_summary_df[
                mode_summary_df["platform"] == platform_label
            ]
            platform_total_by_label[platform_label] = int(
                platform_rows["total_sentence_count"].iloc[0]
            )

        fig = go.Figure()
        for source_bucket in source_order:
            rates = []
            counts = []
            totals = []
            for platform_label in platform_order:
                bucket_rows = mode_summary_df[
                    (mode_summary_df["platform"] == platform_label)
                    & (mode_summary_df["entailment_source_bucket"] == source_bucket)
                ]
                if len(bucket_rows) == 0:
                    rates.append(0.0)
                    counts.append(0)
                    totals.append(platform_total_by_label.get(platform_label, 0))
                else:
                    rates.append(float(bucket_rows["sentence_rate"].iloc[0]))
                    counts.append(int(bucket_rows["sentence_count"].iloc[0]))
                    totals.append(int(bucket_rows["total_sentence_count"].iloc[0]))
            fig.add_trace(
                go.Bar(
                    x=platform_order,
                    y=rates,
                    name=source_bucket,
                    marker_color=color_map[source_bucket],
                    text=[
                        f"{rate:.1%}"
                        if rate > 0
                        else ""
                        for rate in rates
                    ],
                    textposition="inside",
                    textfont=dict(color="white"),
                    customdata=np.column_stack([counts, totals]),
                    hovertemplate=(
                        "Platform: %{x}<br>"
                        "Source bucket: %{fullData.name}<br>"
                        "Sentence rate: %{y:.1%}<br>"
                        "Count: %{customdata[0]} / %{customdata[1]}"
                        "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            barmode="stack",
            xaxis_title="Platform",
            yaxis_title="Rate of Response Claims",
            legend_title="Source",
            title=mode_title,
        )
        fig.update_xaxes(categoryorder="array", categoryarray=platform_order)
        fig.update_yaxes(range=[0, 1], tickformat=".0%")

        fig.write_html(f"{output_dir}/{mode_file_name}.html")
        try:
            paper_fig = with_paper_style(
                fig,
                config=styler(24, 22),
                legend_pos=(0.8, 1.5),
            )
            paper_fig.write_image(f"{output_dir}/{mode_file_name}.pdf", format="pdf")
        except Exception as e:
            logger.warning("Could not write sentence-based NLI PDF: %s", e)

        summary_frames.append(mode_summary_df.assign(plot_file_name=mode_file_name))

    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)


def plot_response_source_nli_sentence_based_judge(
    output_base=None,
    file_name=None,
    chunking_method="citation_marker",
    claim_selection_mode="latest_preceding",
    source_text_mode="full_url_content",
):
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    source_text_mode = _normalize_source_text_mode(source_text_mode)
    if output_base is None and chunking_method != "claim":
        output_base = _response_source_nli_output_base(
            "judge",
            chunking_method,
            claim_selection_mode=claim_selection_mode,
            source_text_mode=source_text_mode,
        )
    if file_name is None:
        file_name = "response_source_nli_sentence_based_judge_summary"
        if chunking_method != "citation_marker":
            file_name = f"{file_name}_{chunking_method}"
        if source_text_mode != "full_url_content":
            file_name = f"{file_name}_{source_text_mode}"
    return plot_response_source_nli_sentence_based(
        output_base=output_base,
        file_name=file_name,
        nli_method="judge",
        chunking_method=chunking_method,
        claim_selection_mode=claim_selection_mode,
        source_text_mode=source_text_mode,
    )


def plot_response_source_nli_entailment_score_boxplot(
    nli_method="bert",
    chunking_method="citation_marker",
    output_base=None,
    file_name=None,
    platform="chatgpt",
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")

    chunking_method = _normalize_chunking_method(chunking_method)
    output_base = output_base or _response_source_nli_output_base(
        nli_method,
        chunking_method,
        platform=platform,
    )
    chunking_label = "chunk" if chunking_method == "citation_marker" else "claim"
    if file_name is None:
        file_name = (
            f"response_source_nli_{nli_method}_{chunking_label}_"
            "entailment_score_boxplot"
        )

    sentence_df = _load_response_source_nli_sentence_based(output_base=output_base)
    if "chunking_method" in sentence_df.columns:
        sentence_df = sentence_df[
            sentence_df["chunking_method"].fillna("citation_marker") == chunking_method
        ].copy()
    if len(sentence_df) == 0:
        return pd.DataFrame()

    source_order = [
        "Marked Citations",
        "Other Citations",
        "Retrieved Sources",
        "Unexplained",
    ]
    color_map = {
        "Marked Citations": "#EF553B",
        "Other Citations": "#AB63FA",
        "Retrieved Sources": "#636EFA",
        "Unexplained": "#00CC96",
    }

    source_buckets = (
        sentence_df["entailment_source_bucket"]
        .fillna("Unexplained")
        .replace(
            {
                "": "Unexplained",
                "Unknown": "Unexplained",
                "unknown": "Unexplained",
            }
        )
    )
    score_col = "bert_nli_confidence" if nli_method == "bert" else "judge_nli_score"
    score_series = pd.to_numeric(sentence_df.get(score_col), errors="coerce")
    plot_df = pd.DataFrame(
        {
            "entailment_source_bucket": source_buckets,
            "entailment_score": score_series,
        }
    )
    plot_df = plot_df[plot_df["entailment_source_bucket"].isin(source_order)].copy()
    plot_df = plot_df.dropna(subset=["entailment_score"]).copy()
    if len(plot_df) == 0:
        return pd.DataFrame()

    summary_df = (
        plot_df.groupby("entailment_source_bucket")["entailment_score"]
        .agg(["count", "mean", "median"])
        .rename(
            columns={
                "count": "score_count",
                "mean": "score_mean",
                "median": "score_median",
            }
        )
        .reindex(source_order)
        .fillna(0)
        .reset_index()
    )

    output_dir = f"{OUTPUT_PATH}/{platform}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(f"{output_dir}/{file_name}.csv", index=False)

    fig = go.Figure()
    for source_bucket in source_order:
        subset = plot_df[plot_df["entailment_source_bucket"] == source_bucket].copy()
        if len(subset) == 0:
            continue
        fig.add_trace(
            go.Box(
                x=[source_bucket] * len(subset),
                y=subset["entailment_score"],
                name=source_bucket,
                legendgroup=source_bucket,
                marker_color=color_map[source_bucket],
                boxmean=True,
                showlegend=False,
                hovertemplate="%{x}<br>Score: %{y:.3f}<extra></extra>",
            )
        )

    score_label = (
        "BERT Entailment Confidence" if nli_method == "bert" else "Judge Entailment Score"
    )
    fig.update_layout(
        xaxis_title="Source Bucket",
        yaxis_title=score_label,
        boxmode="group",
        showlegend=False,
    )
    if nli_method == "bert":
        fig.update_yaxes(range=[0, 1])
    else:
        fig.update_yaxes(range=[0, 5])

    fig.write_html(f"{output_dir}/{file_name}.html")
    try:
        paper_fig = with_paper_style(fig, config=styler(18, 18))
        paper_fig.update_xaxes(tickfont=dict(size=14))
        paper_fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")
    except Exception as e:
        logger.warning("Could not write entailment score boxplot PDF: %s", e)

    return summary_df


def plot_response_source_nli_entailment_score_boxplots_all(platform="chatgpt"):
    combinations = [
        ("bert", "citation_marker"),
        ("bert", "claim"),
        ("judge", "citation_marker"),
        ("judge", "claim"),
    ]
    summary_frames = []

    for nli_method, chunking_method in combinations:
        chunking_label = "chunk" if chunking_method == "citation_marker" else "claim"
        file_name = (
            f"response_source_nli_{nli_method}_{chunking_label}_"
            "entailment_score_boxplot"
        )
        try:
            summary_df = plot_response_source_nli_entailment_score_boxplot(
                nli_method=nli_method,
                chunking_method=chunking_method,
                file_name=file_name,
                platform=platform,
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s/%s plot: %s", nli_method, chunking_method, e)
            continue

        if len(summary_df) == 0:
            continue
        summary_df = summary_df.copy()
        summary_df["nli_method"] = nli_method
        summary_df["chunking_method"] = chunking_method
        summary_frames.append(summary_df)

    if not summary_frames:
        return pd.DataFrame()

    combined_summary_df = pd.concat(summary_frames, ignore_index=True)
    output_dir = f"{OUTPUT_PATH}/{platform}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    combined_summary_df.to_csv(
        f"{output_dir}/response_source_nli_entailment_score_boxplot_all_summary.csv",
        index=False,
    )
    return combined_summary_df

def _smoke_test():
    """Standalone sanity check: score one hardcoded entailing pair and one
    hardcoded contradicting pair, and print the judgments. Needs
    OPENAI_API_KEY; doesn't touch outputs/.
    """
    entailing = compute_nli_scores(
        premise="The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
        hypothesis="The Eiffel Tower is 330 meters tall.",
    )
    print("Entailing pair judgment:", entailing)

    contradicting = compute_nli_scores(
        premise="The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
        hypothesis="The Eiffel Tower was built in 1920 and is only 50 meters tall.",
    )
    print("Contradicting pair judgment:", contradicting)


if __name__ == "__main__":
    _smoke_test()
