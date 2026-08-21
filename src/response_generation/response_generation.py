"""§5.2 analyses: how responses are grounded in (or ungrounded from) their
cited/retrieved sources -- scraping cited URLs' actual content, computing
NLI entailment between response claims and that content, and factuality
scoring by grounding source (associated citation / other citation / search
result / parametric knowledge).

Same scope note as query_reformulations.py / source_selection.py: written
for the paper's full cohort, organized as a library of individually-
runnable analysis functions (see the __main__ call list), each writing its
own figure/table under outputs/response_generation/.

Pipeline dependency: extract_response_and_sources(web_df) (and
extract_response_and_sources_other_platforms() for non-ChatGPT platforms)
writes outputs/[<platform>/]metadata/response_and_sources.pkl -- most of
the grounding/NLI functions here read it, and it's also the prerequisite
source_selection.py's count_unique_retrieved_safe_cited() and related
functions need but don't produce themselves. Run it before those.
"""

import os
import json
import re
import ast
import hashlib
import logging
from collections import Counter
from tqdm import tqdm
import pandas as pd
import requests
from urllib.parse import urlparse, unquote
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import asyncio
import fitz
from playwright.async_api import async_playwright
from readability import Document
from rouge_score import rouge_scorer
import numpy as np
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.prompts.evaluator_prompts import SYSTEM_PROMPT_RESP_SYNT, USER_PROMPT_RESP_SYNT, SYSTEM_PROMPT_CLAIM_EXTRACTION, USER_PROMPT_CLAIM_EXTRACTION, SYSTEM_PROMPT_CLAIM_FACTUALITY_EVAL, USER_PROMPT_CLAIM_FACTUALITY_EVAL


load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

pio.defaults.mathjax = None
from src.utils.common_io import *
from src.utils.chatgpt_conversation_utils import *
from src.utils.figure_style import with_paper_style, styler
from src.web_search_decision.extraction import load_web_data_from_file

CONF = "./response_generation"


def _platform_metadata_dir(platform):
    """outputs/metadata for chatgpt, outputs/<platform>/metadata otherwise
    (same convention as src.web_search_decision.extraction._metadata_dir).
    Used by extract_response_and_sources[_other_platforms] so Claude/Grok/
    DeepSeek runs don't overwrite ChatGPT's response_and_sources.pkl (and
    each other's) by all writing to the same flat path."""
    if platform == "chatgpt":
        return f"{OUTPUT_PATH}/metadata"
    return f"{OUTPUT_PATH}/{platform}/metadata"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DIRECT_API_DOMAINS = {"wikipedia.org"}
SKIP_REQUESTS_DOMAINS = {"politico.com", "reuters.com"}
REQUEST_TIMEOUT = int(os.getenv("ARTICLE_REQUEST_TIMEOUT"))
WIKIPEDIA_TIMEOUT = int(os.getenv("ARTICLE_WIKIPEDIA_TIMEOUT"))
PLAYWRIGHT_GOTO_TIMEOUT = int(os.getenv("ARTICLE_PLAYWRIGHT_GOTO_TIMEOUT"))
PLAYWRIGHT_NETWORKIDLE_TIMEOUT = int(
    os.getenv("ARTICLE_PLAYWRIGHT_NETWORKIDLE_TIMEOUT")
)
PLAYWRIGHT_FALLBACK_TIMEOUT = float(
    os.getenv("ARTICLE_PLAYWRIGHT_FALLBACK_TIMEOUT")
)
URL_FETCH_TIMEOUT = float(os.getenv("ARTICLE_URL_FETCH_TIMEOUT"))
URL_FETCH_CHECKPOINT_EVERY = int(os.getenv("ARTICLE_URL_CHECKPOINT_EVERY"))
RESPONSE_URLS_CONTENT_PATH = (
    f"{OUTPUT_PATH}/metadata/response_and_sources_url_content.json"
)
RESPONSE_SOURCE_EFFECT_EVALUATIONS_BASE = (
    f"{OUTPUT_PATH}/metadata/response_source_effect_evaluations"
)
RESPONSE_SOURCE_NLI_SENTENCE_BASED_JUDGE_BASE = (
    f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge"
)
RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE = (
    f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_bert"
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
    f"{OUTPUT_PATH}/metadata/cited_url_validity_labels.json"
)
BERT_NLI_MODEL_NAME = os.getenv("BERT_NLI_MODEL_NAME")
BERT_NLI_MAX_LENGTH = int(os.getenv("BERT_NLI_MAX_LENGTH"))
NLI_JUDGE_CONTEXT_WINDOW_TOKENS = 128000
NLI_JUDGE_MAX_OUTPUT_TOKENS = 256
NLI_JUDGE_TOKEN_SAFETY_MARGIN = 2000
NLI_ESTIMATED_CHARS_PER_TOKEN = 3.0
CLAIM_EXTRACTION_MODEL = os.getenv("CLAIM_EXTRACTION_MODEL")
CLAIM_EXTRACTION_MAX_INPUT_CHARS = int(os.getenv("CLAIM_EXTRACTION_MAX_INPUT_CHARS"))
CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS = int(
    os.getenv("CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS")
)
CLAIM_EXTRACTION_CACHE_PATH = os.getenv("CLAIM_EXTRACTION_CACHE_PATH")
FACTUALITY_JUDGE_MODEL = os.getenv("FACTUALITY_JUDGE_MODEL")
FACTUALITY_JUDGE_MAX_OUTPUT_TOKENS = int(
    os.getenv("FACTUALITY_JUDGE_MAX_OUTPUT_TOKENS")
)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def find_similarity(page_content, response):
    embeddings = model.encode([response, page_content])

    response_emb = embeddings[0]
    page_content_embs = embeddings[1:]

    scores = cosine_similarity(
        response_emb.reshape(1, -1), page_content_embs.reshape(1, -1)
    )[0]

    return float(scores.mean())

def extract_first_json_object(text):
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


def _coerce_claim_list(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["claims", "claim_list", "items", "sentences", "chunks"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _clean_claims(claims):
    cleaned_claims = []
    seen = set()
    for claim in claims:
        if isinstance(claim, dict):
            claim = (
                claim.get("claim")
                or claim.get("text")
                or claim.get("statement")
                or ""
            )
        claim = str(claim or "").strip(" -*\t\n")
        if len(claim) < 8 or not re.search(r"[A-Za-z]", claim):
            continue
        if claim in seen:
            continue
        seen.add(claim)
        cleaned_claims.append(claim)
    return cleaned_claims


def _normalize_claim_cache_entries(value):
    if not isinstance(value, list):
        return []

    normalized_entries = []
    seen = set()
    for item in value:
        user_query = ""
        claim_value = item
        if isinstance(item, dict):
            user_query = str(item.get("user_query", "") or "").strip()
            claim_value = (
                item.get("claim")
                or item.get("text")
                or item.get("statement")
                or ""
            )
        cleaned_claims = _clean_claims([claim_value])
        if not cleaned_claims:
            continue
        claim_text = cleaned_claims[0]
        dedupe_key = (claim_text, user_query)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_entries.append(
            {
                "claim": claim_text,
                "user_query": user_query,
            }
        )
    return normalized_entries


def _claim_cache_claim_texts(entries):
    return _clean_claims(entries if isinstance(entries, list) else [])


def _claim_cache_key(text):
    text = str(text or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_claims_cache(cache_path=CLAIM_EXTRACTION_CACHE_PATH):
    if not cache_path or not os.path.exists(cache_path):
        return {}

    raw_cache = load_json(cache_path)
    if not isinstance(raw_cache, dict):
        logger.warning("Claim cache at %s is not a JSON object; ignoring.", cache_path)
        return {}

    normalized_cache = {}
    for key, value in raw_cache.items():
        normalized_entries = _normalize_claim_cache_entries(value)
        if normalized_entries:
            normalized_cache[str(key)] = normalized_entries
    return normalized_cache


def _save_claims_cache(claims_cache, cache_path=CLAIM_EXTRACTION_CACHE_PATH):
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    to_json(claims_cache, cache_path, indent=2)


def _claim_cache_factuality_output_path(
    cache_path=CLAIM_EXTRACTION_CACHE_PATH,
    output_suffix="_factuality",
):
    if not cache_path:
        raise ValueError("cache_path must be provided")
    base, ext = os.path.splitext(cache_path)
    ext = ext or ".json"
    return f"{base}{output_suffix}{ext}"


def _extract_user_query_from_turn_msgs(turn_msgs_value):
    if not turn_msgs_value:
        return ""
    turn_msgs = turn_msgs_value
    if isinstance(turn_msgs_value, str):
        try:
            turn_msgs = json.loads(turn_msgs_value)
        except Exception:
            return ""
    if not isinstance(turn_msgs, list):
        return ""

    for msg in turn_msgs:
        if not isinstance(msg, dict):
            continue
        author = msg.get("author", {})
        role = str(author.get("name") or author.get("role") or "").strip()
        if role != "user":
            continue
        parts = msg.get("content", {}).get("parts", [])
        if isinstance(parts, list):
            return " ".join([str(part) for part in parts]).strip()
    return ""


def _row_user_query(row):
    user_msg_history = row.get("user_msg_history", [])
    if isinstance(user_msg_history, str) and user_msg_history.strip():
        try:
            parsed_history = ast.literal_eval(user_msg_history)
            if isinstance(parsed_history, list) and parsed_history:
                return str(parsed_history[-1] or "").strip()
        except (ValueError, SyntaxError):
            return user_msg_history.strip()
    if isinstance(user_msg_history, list) and user_msg_history:
        return str(user_msg_history[-1] or "").strip()

    return _extract_user_query_from_turn_msgs(row.get("turn_msgs"))

def _claim_cache_chunk_texts_from_response(response_text):
    response_text = str(response_text or "")
    if not response_text:
        return []

    citation_marker_pattern = re.compile(
        r"\ue200(?=[^\ue201]*\ue202[A-Za-z]+\d+[A-Za-z]+\d+(?:\ue202|\ue201))[^\ue201]*\ue201"
    )
    marker_matches = list(citation_marker_pattern.finditer(response_text))

    chunk_texts = []
    previous_end = 0
    for marker_match in marker_matches:
        raw_chunk = str(response_text[previous_end:marker_match.start()] or "").strip()
        if raw_chunk:
            chunk_texts.append(raw_chunk)
        previous_end = marker_match.end()

    tail_chunk = str(response_text[previous_end:] or "").strip()
    if tail_chunk:
        chunk_texts.append(tail_chunk)

    if not chunk_texts:
        whole_text = response_text.strip()
        if whole_text:
            chunk_texts.append(whole_text)

    return chunk_texts


def _build_claim_cache_user_query_lookup():
    response_df = _load_response_source_similarity_input().copy()
    query_counts_by_cache_key = {}
    for _, row in response_df.iterrows():
        user_query = _row_user_query(row)
        if not user_query:
            continue
        response_text = str(row.get("asistant_response", "") or "")
        for chunk_text in _claim_cache_chunk_texts_from_response(response_text):
            cache_key = _claim_cache_key(chunk_text)
            counts = query_counts_by_cache_key.setdefault(cache_key, {})
            counts[user_query] = counts.get(user_query, 0) + 1

    cache_key_to_user_query = {}
    for cache_key, counts in query_counts_by_cache_key.items():
        if not counts:
            continue
        cache_key_to_user_query[cache_key] = max(
            counts.items(),
            key=lambda item: (item[1], len(item[0])),
        )[0]
    return cache_key_to_user_query


def _record_join_key(record):
    return tuple(
        str(record.get(col, "") or "").strip()
        for col in ["user_id", "conv_id", "turn_id"]
    )


def _build_response_source_user_query_lookup():
    response_df = _load_response_source_similarity_input().copy()
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


def extract_claims_from_text(text):
    text = str(text or "").strip()
    if not text:
        return []

    prompt_text = text[:CLAIM_EXTRACTION_MAX_INPUT_CHARS]
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT_CLAIM_EXTRACTION},
        {
            "role": "user",
            "content": USER_PROMPT_CLAIM_EXTRACTION.format(text=prompt_text),
        },
    ]

    response_text = ""
    try:
        response = client.chat.completions.create(
            model=CLAIM_EXTRACTION_MODEL,
            messages=msg,
            max_tokens=CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS,
            temperature=0.0,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Claim extraction failed: %s", e)
        return []

    parsed_payload = None
    try:
        parsed_payload = json.loads(response_text)
    except Exception:
        parsed_payload = extract_first_json_object(response_text)

    claims = _coerce_claim_list(parsed_payload)
    return _clean_claims(claims)


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


def extract_response_and_sources(web_df):
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
    web_df["asistant_response"] = [""] * len(web_df)
    web_df["user_query"] = [""] * len(web_df)

    for i, row in tqdm(web_df.iterrows()):
        msgs = json.loads(row["turn_msgs"])
        parts = msgs[-1].get("content", {}).get("parts", [])
        asistant_response = " ".join([str(p) for p in parts])
        user_query = ""
        srcs_retrieved = []
        srcs_safe_urls = []
        srcs_cited = []
        for msg in msgs:
            author = msg.get("author", {})
            role_ = author.get("name") or author.get("role", "")
            if not user_query and role_ == "user":
                user_parts = msg.get("content", {}).get("parts", [])
                user_query = " ".join([str(p) for p in user_parts]).strip()

            # retrieved
            retrieved = msg.get("metadata", {}).get("search_result_groups", [])
            for r in retrieved:
                entries = r.get("entries", [])
                for entry in entries:
                    url = entry.get("url", "")
                    if url:
                        d = urlparse(entry["url"]).netloc.replace("www.", "")
                        srcs_retrieved.append(
                            {
                                "url": url,
                                "domain": d,
                                "title": entry.get("title", ""),
                                "ref_index": (
                                    entry.get("ref_id", {}).get("ref_index", None)
                                    if entry.get("ref_id", {})
                                    else None
                                ),
                                "turn_index": (
                                    entry.get("ref_id", {}).get("turn_index", None)
                                    if entry.get("ref_id", {})
                                    else None
                                ),
                                "snippet": entry["snippet"],
                            }
                        )

            retrieved = msg.get("metadata", {}).get("image_results", [])
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
            safe_urls = msg.get("metadata", {}).get("safe_urls", [])
            for r in safe_urls:
                if r:
                    url = r.removesuffix("?utm_source=chatgpt.com").removesuffix(
                        "&utm_source=chatgpt.com"
                    )
                    d = urlparse(url).netloc.replace("www.", "")
                    srcs_safe_urls.append({"url": url, "domain": d})

            # cited
            cited = msg.get("metadata", {}).get("content_references", [])
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

                        url = r.get("url", "")
                        if url:
                            url = url.removesuffix(
                                "?utm_source=chatgpt.com"
                            ).removesuffix("&utm_source=chatgpt.com")
                            d = urlparse(url).netloc.replace("www.", "")
                            srcs_cited.append(
                                {
                                    "url": url,
                                    "domain": d,
                                    "title": r.get("title"),
                                    "snippet": r.get("snippet"),
                                    "ref_index": cited_ranks[0],
                                    "turn_index": cited_turns[0],
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
                                    url = (
                                        item.get("url", "")
                                        .removesuffix("?utm_source=chatgpt.com")
                                        .removesuffix("&utm_source=chatgpt.com")
                                    )
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
                                                "turn_index": ref.get(
                                                    "turn_index", None
                                                ),
                                            }
                                        )

        web_df.at[i, "srcs_retrieved"] = srcs_retrieved
        web_df.at[i, "srcs_safe_urls"] = srcs_safe_urls
        web_df.at[i, "srcs_cited"] = _dedupe_cited_items(srcs_cited)
        web_df.at[i, "asistant_response"] = asistant_response
        web_df.at[i, "user_query"] = user_query

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)

    platform_dir = _platform_metadata_dir("chatgpt")
    os.makedirs(platform_dir, exist_ok=True)
    web_df.to_csv(
        f"{platform_dir}/response_and_sources.csv",
        index=False,
    )
    web_df.to_pickle(f"{platform_dir}/response_and_sources.pkl")


# ============================================================
# _other_platforms: Ported from our internal repo (claude/grok/deepseek).
# Public entry is `extract_response_and_sources_other_platforms(web_df, platform)`.
# Per-platform helpers are prefixed with `_` to mark as internal.
# ============================================================


def extract_response_and_sources_other_platforms(web_df, platform):
    """Dispatcher for non-chatgpt platforms."""
    if platform == "claude":
        return _extract_response_and_sources_claude(web_df)
    if platform == "grok":
        return _extract_response_and_sources_grok(web_df)
    if platform == "deepseek":
        return _extract_response_and_sources_deepseek(web_df)
    raise ValueError(
        f"Unknown platform: {platform!r}. Use 'claude', 'grok', or 'deepseek'."
    )


def _extract_response_and_sources_claude(web_df):
    """Claude has no `safe_url` concept. srcs_cited from Anthropic Citations API
    (text block `citations`). srcs_retrieved from web_search/web_fetch tool_result."""
    web_df["srcs_retrieved"] = [[]] * len(web_df)
    web_df["srcs_safe_urls"] = [[]] * len(web_df)
    web_df["srcs_cited"] = [[]] * len(web_df)
    web_df["asistant_response"] = [""] * len(web_df)
    web_df["user_query"] = [""] * len(web_df)

    for i, row in tqdm(web_df.iterrows(), total=len(web_df)):
        msgs = json.loads(row["turn_msgs"])
        retrieved = []
        cited = []
        response_parts = []
        user_query = ""

        for msg in msgs:
            if msg.get("sender") != "assistant":
                if not user_query and msg.get("sender") == "human":
                    user_parts = [
                        b.get("text", "") for b in (msg.get("content") or [])
                        if b.get("type") == "text" and b.get("text")
                    ]
                    user_query = " ".join(user_parts).strip() or str(msg.get("text", "")).strip()
                continue
            for b in msg.get("content") or []:
                btype = b.get("type")

                if btype == "text":
                    text = b.get("text", "")
                    if text:
                        response_parts.append(text)
                    for cit in b.get("citations") or []:
                        if not isinstance(cit, dict):
                            continue
                        details = cit.get("details") or {}
                        url = (details.get("url") or cit.get("url") or "").strip()
                        if not url:
                            continue
                        start = cit.get("start_index")
                        end = cit.get("end_index")
                        cited_text = ""
                        if (
                            isinstance(start, int)
                            and isinstance(end, int)
                            and 0 <= start < end <= len(text)
                        ):
                            cited_text = text[start:end]
                        cited.append({
                            "url": url,
                            "domain": urlparse(url).netloc.replace("www.", ""),
                            "title": details.get("title") or cit.get("title") or "",
                            "cited_text": cited_text,
                            "start_index": start,
                            "end_index": end,
                            "citation_type": details.get("type", ""),
                        })

                elif btype == "tool_result" and b.get("name") in ("web_search", "web_fetch"):
                    content_items = b.get("content")
                    if not isinstance(content_items, list):
                        continue
                    for item in content_items:
                        if not isinstance(item, dict):
                            continue
                        url = (item.get("url") or "").strip()
                        if not url:
                            continue
                        retrieved.append({
                            "url": url,
                            "domain": urlparse(url).netloc.replace("www.", ""),
                            "title": item.get("title", "") or "",
                            "snippet": item.get("text", "") or "",
                        })

        web_df.at[i, "srcs_retrieved"] = retrieved
        web_df.at[i, "srcs_safe_urls"] = []
        web_df.at[i, "srcs_cited"] = cited
        web_df.at[i, "asistant_response"] = " ".join(response_parts).strip()
        web_df.at[i, "user_query"] = user_query

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)
    platform_dir = _platform_metadata_dir("claude")
    os.makedirs(platform_dir, exist_ok=True)
    web_df.to_csv(f"{platform_dir}/response_and_sources.csv", index=False)
    web_df.to_pickle(f"{platform_dir}/response_and_sources.pkl")


def _extract_response_and_sources_grok(web_df):
    """Grok — srcs_retrieved from response-level `web_search_results` (the
    'N sources' panel). srcs_cited from `card_attachments_json` entries with
    cardType == "citation_card" (keeps `card_id` for downstream mapping)."""
    any_render_tag_pattern = re.compile(r"<grok:render\b[^>]*>.*?</grok:render>", re.DOTALL)

    def _parse_card(raw):
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    web_df["srcs_retrieved"] = [[]] * len(web_df)
    web_df["srcs_safe_urls"] = [[]] * len(web_df)
    web_df["srcs_cited"] = [[]] * len(web_df)
    web_df["asistant_response"] = [""] * len(web_df)
    web_df["user_query"] = [""] * len(web_df)

    for i, row in tqdm(web_df.iterrows(), total=len(web_df)):
        msgs = json.loads(row["turn_msgs"])
        retrieved = []
        cited = []
        response_parts = []
        user_query = ""

        for msg in msgs:
            if str(msg.get("sender", "")).lower() != "assistant":
                if not user_query and str(msg.get("sender", "")).lower() == "human":
                    user_query = str(msg.get("message", "") or "").strip()
                continue

            for r in msg.get("web_search_results") or []:
                if not isinstance(r, dict):
                    continue
                url = (r.get("url") or "").strip()
                if not url:
                    continue
                retrieved.append({
                    "url": url,
                    "domain": urlparse(url).netloc.replace("www.", ""),
                    "title": r.get("title", "") or "",
                    "snippet": r.get("preview", "") or r.get("snippet", "") or "",
                })

            for raw_card in msg.get("card_attachments_json") or []:
                card = _parse_card(raw_card)
                if not card or card.get("cardType") != "citation_card":
                    continue
                url = (card.get("url") or "").strip()
                cited.append({
                    "url": url,
                    "domain": urlparse(url).netloc.replace("www.", "") if url else "",
                    "title": card.get("title", "") or "",
                    "snippet": card.get("preview", "") or card.get("snippet", "") or "",
                    "card_id": card.get("id", ""),
                })

            message_text = msg.get("message", "") or ""
            if message_text.strip():
                response_parts.append(message_text)

        web_df.at[i, "srcs_retrieved"] = retrieved
        web_df.at[i, "srcs_safe_urls"] = []
        web_df.at[i, "srcs_cited"] = cited
        web_df.at[i, "asistant_response"] = " ".join(response_parts).strip()
        web_df.at[i, "user_query"] = user_query

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)
    platform_dir = _platform_metadata_dir("grok")
    os.makedirs(platform_dir, exist_ok=True)
    web_df.to_csv(f"{platform_dir}/response_and_sources.csv", index=False)
    web_df.to_pickle(f"{platform_dir}/response_and_sources.pkl")


def _extract_response_and_sources_deepseek(web_df):
    """DeepSeek — srcs_retrieved from SEARCH fragments (url/title/snippet/cite_index).
    srcs_cited from SEARCH results referenced by `[citation:N]` markers in response text,
    matched on cite_index."""
    citation_marker_pattern = re.compile(r"\[citation:(\d+)\]")

    web_df["srcs_retrieved"] = [[]] * len(web_df)
    web_df["srcs_safe_urls"] = [[]] * len(web_df)
    web_df["srcs_cited"] = [[]] * len(web_df)
    web_df["asistant_response"] = [""] * len(web_df)
    web_df["user_query"] = [""] * len(web_df)

    for i, row in tqdm(web_df.iterrows(), total=len(web_df)):
        msgs = json.loads(row["turn_msgs"])
        retrieved = []
        response_parts = []
        cite_lookup = {}
        user_query = ""

        for node in msgs:
            msg = node.get("message") or {}
            fragments = msg.get("fragments") or []
            files = msg.get("files") or []
            is_user = (
                any(f.get("type") == "REQUEST" for f in fragments)
                or (not fragments and bool(files))
            )
            if is_user:
                if not user_query:
                    request_parts = [
                        f.get("content", "") for f in fragments
                        if f.get("type") == "REQUEST" and f.get("content")
                    ]
                    user_query = " ".join(request_parts).strip()
                continue

            for f in fragments:
                ftype = f.get("type")
                if ftype == "RESPONSE":
                    content = f.get("content") or ""
                    if content:
                        response_parts.append(content)
                elif ftype == "SEARCH":
                    for r in f.get("results") or []:
                        url = (r.get("url") or "").strip()
                        if not url:
                            continue
                        cite_index = r.get("cite_index")
                        entry = {
                            "url": url,
                            "domain": urlparse(url).netloc.replace("www.", ""),
                            "title": r.get("title", "") or "",
                            "snippet": r.get("snippet", "") or "",
                            "ref_index": cite_index,
                        }
                        retrieved.append(entry)
                        if cite_index is not None and cite_index not in cite_lookup:
                            cite_lookup[cite_index] = entry

        asistant_response = "\n".join(response_parts).strip()

        cited = []
        seen_indices = set()
        for m in citation_marker_pattern.finditer(asistant_response):
            try:
                idx = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            src = cite_lookup.get(idx)
            if src is None:
                cited.append({
                    "url": "", "domain": "", "title": "", "snippet": "",
                    "ref_index": idx,
                })
            else:
                cited.append(dict(src))

        web_df.at[i, "srcs_retrieved"] = retrieved
        web_df.at[i, "srcs_safe_urls"] = []
        web_df.at[i, "srcs_cited"] = cited
        web_df.at[i, "asistant_response"] = asistant_response
        web_df.at[i, "user_query"] = user_query

    web_df.drop(columns=["turn_msgs"], inplace=True)
    web_df.reset_index(drop=True, inplace=True)
    platform_dir = _platform_metadata_dir("deepseek")
    os.makedirs(platform_dir, exist_ok=True)
    web_df.to_csv(f"{platform_dir}/response_and_sources.csv", index=False)
    web_df.to_pickle(f"{platform_dir}/response_and_sources.pkl")


def clean_html_for_readability(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text


def extract_clean_text_from_html(html):
    html = clean_html_for_readability(html)
    if not html:
        return ""

    try:
        doc = Document(html)
        clean_html = doc.summary()
    except Exception:
        clean_html = html

    soup = BeautifulSoup(clean_html, "html.parser")
    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)

    if len(clean_text.strip()) < 200:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)

    return clean_text


def get_article_text(url):
    logger.info("Fetching URL with requests: %s", url)
    session = requests.Session()
    session.headers.update(HEADERS)

    response = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    content_type = (response.headers.get("content-type") or "").lower()

    if (
        "application/pdf" in content_type
        or url.lower().endswith(".pdf")
        or "/bitstream/" in url.lower()
        or response.content[:4] == b"%PDF"
    ):
        logger.info("Detected PDF content from requests path: %s", url)
        return extract_text_from_pdf_bytes(response.content)

    response.encoding = response.encoding or response.apparent_encoding
    return extract_clean_text_from_html(response.text)


def get_article_text_wikipedia(url):
    logger.info("Fetching URL with Wikipedia API: %s", url)
    parsed = urlparse(url)
    title = unquote(parsed.path.removeprefix("/wiki/")).strip()
    if not title:
        raise ValueError(f"Could not parse Wikipedia title from URL: {url}")

    api_url = f"{parsed.scheme}://{parsed.netloc}/w/api.php"
    response = requests.get(
        api_url,
        headers=HEADERS,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
        timeout=WIKIPEDIA_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "").strip()
        if extract:
            return extract
    raise ValueError(f"Wikipedia API returned no extract for {url}")


def get_domain(url):
    return urlparse(url).netloc.lower().replace("www.", "")


def extract_text_from_pdf_bytes(pdf_bytes):
    if not pdf_bytes:
        return ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.warning("Failed to open PDF bytes with PyMuPDF: %s", e)
        return ""

    try:
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    finally:
        doc.close()


async def fetch_url_content(url, browser=None, url_cache=None):
    if url_cache is not None and url in url_cache:
        logger.info("URL cache hit: %s", url)
        return url_cache[url]

    domain = get_domain(url)

    if any(domain.endswith(suffix) for suffix in DIRECT_API_DOMAINS):
        try:
            content = await asyncio.to_thread(get_article_text_wikipedia, url)
            logger.info("Wikipedia API path succeeded: %s", url)
            if url_cache is not None:
                url_cache[url] = content
            return content
        except Exception as e:
            logger.warning("Wikipedia API path failed for %s: %s", url, e)

    if not any(domain.endswith(suffix) for suffix in SKIP_REQUESTS_DOMAINS):
        try:
            content = await asyncio.to_thread(get_article_text, url)
            logger.info("Requests path succeeded: %s", url)
            if url_cache is not None:
                url_cache[url] = content
            return content
        except Exception as e:
            logger.warning("Requests path failed for %s: %s", url, e)
    else:
        logger.info("Skipping requests fast path for domain %s: %s", domain, url)

    try:
        content = await asyncio.wait_for(
            get_article_text_planB(url, browser=browser),
            timeout=PLAYWRIGHT_FALLBACK_TIMEOUT,
        )
        logger.info("Playwright path succeeded: %s", url)
        if url_cache is not None:
            url_cache[url] = content
        return content
    except asyncio.TimeoutError:
        logger.warning(
            "Playwright path timed out after %.1fs for %s",
            PLAYWRIGHT_FALLBACK_TIMEOUT,
            url,
        )
    except Exception as e:
        logger.warning("Playwright path failed for %s: %s", url, e)

    logger.warning("All extraction paths failed for %s", url)
    if url_cache is not None:
        url_cache[url] = ""
    return ""


COOKIE_BUTTON_TEXTS: List[str] = [
    "accept",
    "accept all",
    "agree",
    "agree to all",
    "allow all",
    "allow cookies",
    "consent",
    "continue",
    "i agree",
    "ok",
    "okay",
]

PAYWALL_BUTTON_TEXTS: List[str] = [
    "continue reading",
    "no thanks",
    "not now",
    "close",
    "dismiss",
    "maybe later",
]


async def accept_cookie_banners(page):
    # Try a few broad strategies because cookie walls vary heavily across sites.
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button[aria-label*='Accept' i]",
        "button[title*='Accept' i]",
        "[id*='accept' i]",
        "[class*='accept' i]",
        "[data-testid*='accept' i]",
        "[data-test*='accept' i]",
    ]

    for frame in page.frames:
        for selector in selectors:
            try:
                locator = frame.locator(selector).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                pass

        for text in COOKIE_BUTTON_TEXTS:
            try:
                locator = frame.get_by_role(
                    "button", name=re.compile(rf"^{re.escape(text)}$", re.I)
                ).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                pass

            try:
                locator = frame.get_by_text(re.compile(rf"\b{re.escape(text)}\b", re.I)).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                pass


async def dismiss_paywall_overlays(page):
    selectors = [
        "[aria-label='Close']",
        "button[aria-label*='close' i]",
        "[data-testid*='close' i]",
        "[class*='close' i]",
        "[class*='modal' i]",
        "[class*='overlay' i]",
        "[class*='paywall' i]",
        "[id*='modal' i]",
        "[id*='overlay' i]",
        "[id*='paywall' i]",
    ]

    for frame in page.frames:
        for text in PAYWALL_BUTTON_TEXTS:
            try:
                locator = frame.get_by_role(
                    "button", name=re.compile(rf"\b{re.escape(text)}\b", re.I)
                ).first
                if await locator.is_visible(timeout=1000):
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(1000)
                    return
            except Exception:
                pass

        for selector in selectors:
            try:
                locator = frame.locator(selector).first
                if await locator.is_visible(timeout=1000):
                    await locator.evaluate(
                        """node => {
                            node.remove();
                            document.body.style.overflow = 'auto';
                            document.documentElement.style.overflow = 'auto';
                        }"""
                    )
                await page.wait_for_timeout(500)
            except Exception:
                pass

    try:
        await page.evaluate(
            """
            () => {
                const patterns = /(paywall|gateway|modal|overlay|subscribe)/i;
                for (const node of Array.from(document.querySelectorAll('div,section,aside'))) {
                    const attrs = [node.id || '', node.className || '', node.getAttribute('data-testid') || ''].join(' ');
                    if (patterns.test(attrs)) {
                        node.remove();
                    }
                }
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';
            }
            """
        )
    except Exception:
        pass


async def extract_text_from_live_dom(page):
    article_selectors = [
        "article",
        "main article",
        "[data-testid='ArticleBodyWrapper']",
        "[data-testid*='article-body' i]",
        "[class*='article-body' i]",
        "[class*='ArticleBody' i]",
        "main",
    ]

    for selector in article_selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0 and await locator.is_visible(timeout=1000):
                text = await locator.inner_text(timeout=3000)
                if text and len(text.strip()) > 300:
                    logger.info("Extracted content from live DOM selector %s", selector)
                    return text.strip()
        except Exception:
            pass

    return ""


async def download_pdf_text(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Capture the main response
        response = await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        if response is None:
            await browser.close()
            raise ValueError("No response")

        content_type = response.headers.get("content-type", "")

        if "application/pdf" not in content_type:
            await browser.close()
            raise ValueError(f"Blocked or not PDF. Content-Type: {content_type}")

        pdf_bytes = await response.body()
        await browser.close()

    return extract_text_from_pdf_bytes(pdf_bytes)


async def get_article_text_planB(url, browser=None):
    logger.info("Fetching URL with Playwright fallback: %s", url)
    if ".pdf" in url.lower() or "/bitstream/" in url.lower():
        return await download_pdf_text(url)

    if browser is None:
        async with async_playwright() as p:
            owned_browser = await p.chromium.launch(
                headless=True, args=["--disable-blink-features=AutomationControlled"]
            )
            try:
                return await get_article_text_planB(url, browser=owned_browser)
            finally:
                await owned_browser.close()

    context = await browser.new_context(
        user_agent=HEADERS["User-Agent"],
        locale="en-US",
        extra_http_headers=HEADERS,
        java_script_enabled=True,
        ignore_https_errors=True,
        viewport={"width": 1440, "height": 1600},
    )

    try:
        page = await context.new_page()
        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            """
        )

        await page.goto(
            url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_GOTO_TIMEOUT
        )
        await page.wait_for_timeout(1000)

        try:
            await accept_cookie_banners(page)
        except Exception:
            pass

        try:
            await dismiss_paywall_overlays(page)
        except Exception:
            pass

        try:
            await page.wait_for_load_state(
                "networkidle", timeout=PLAYWRIGHT_NETWORKIDLE_TIMEOUT
            )
        except Exception:
            pass

        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(500)
        except Exception:
            pass

        live_text = await extract_text_from_live_dom(page)
        content = await page.content()
    finally:
        await context.close()

    if live_text:
        return live_text

    return extract_clean_text_from_html(content)



def _load_response_source_similarity_input():
    pkl_path = f"{OUTPUT_PATH}/metadata/response_and_sources.pkl"
    csv_path = f"{OUTPUT_PATH}/metadata/response_and_sources.csv"

    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        if not os.path.exists(csv_path):
            raise
        logger.warning(
            "Failed to load %s, falling back to %s: %s",
            pkl_path,
            csv_path,
            e,
        )
        df = pd.read_csv(csv_path)

        def _parse_source_list(value):
            if isinstance(value, list):
                return value
            if not isinstance(value, str) or not value.strip():
                return []
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
            return parsed if isinstance(parsed, list) else []

        for source_col in ["srcs_retrieved", "srcs_safe_urls", "srcs_cited"]:
            if source_col in df.columns:
                df[source_col] = df[source_col].apply(_parse_source_list)

    selected_topics = ["Science", "Health", "Politics & History"]
    random_state = 42
    image_url_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
        ".tif",
        ".tiff",
        ".avif",
        ".heic",
        ".heif",
        ".jfif",
        ".pjpeg",
        ".pjp",
        ".mov"
    }

    def _is_image_url(url):
        if not url:
            return False
        lower_url = url.lower()
        return any(lower_url.endswith(ext) for ext in image_url_extensions)

    def _is_bing_tse_url(url):
        if not url:
            return False
        parsed = urlparse(url.lower())
        host = parsed.netloc or ""
        return (
            parsed.scheme in {"http", "https"}
            and host.startswith("tse")
            and host.endswith(".mm.bing.net")
        )

    def _row_has_cited_or_retrieved_image_url(row):
        for source_col in ["srcs_cited", "srcs_retrieved"]:
            sources = row.get(source_col, [])
            if not isinstance(sources, list):
                continue
            for src in sources:
                if not isinstance(src, dict):
                    continue
                source_url = src.get("url", "")
                if _is_image_url(source_url) or _is_bing_tse_url(source_url):
                    return True
        return False

    df = (
        df[
            (df["language"] == "en")
            & (df["topic"].isin(selected_topics))
        ]
        .copy()
    )
    if "srcs_cited" in df.columns and "srcs_retrieved" in df.columns:
        has_image_url_mask = df.apply(_row_has_cited_or_retrieved_image_url, axis=1)
        df = df.loc[~has_image_url_mask].copy()

    # print(df["topic"].value_counts())

    sampled_frames = []
    for topic in selected_topics:
        topic_df = df[df["topic"] == topic].copy()
        if topic_df.empty:
            continue
        sample_n = min(100, len(topic_df))
        sampled_frames.append(topic_df.sample(n=sample_n, random_state=random_state))

    if not sampled_frames:
        return df.iloc[0:0].reset_index(drop=True)

    sampled_df = (
        pd.concat(sampled_frames, ignore_index=True)
        .sort_values(["topic", "conv_id", "turn_id"], kind="stable")
        .reset_index(drop=True)
    )
    return sampled_df


def _load_response_and_sources_df():
    pkl_path = f"{OUTPUT_PATH}/metadata/response_and_sources.pkl"
    csv_path = f"{OUTPUT_PATH}/metadata/response_and_sources.csv"

    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        if not os.path.exists(csv_path):
            raise
        logger.warning(
            "Failed to load %s, falling back to %s: %s",
            pkl_path,
            csv_path,
            e,
        )
        df = pd.read_csv(csv_path)

        def _parse_source_list(value):
            if isinstance(value, list):
                return value
            if not isinstance(value, str) or not value.strip():
                return []
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
            return parsed if isinstance(parsed, list) else []

        for source_col in ["srcs_retrieved", "srcs_safe_urls", "srcs_cited"]:
            if source_col in df.columns:
                df[source_col] = df[source_col].apply(_parse_source_list)

    return df.copy()


def _iter_response_source_urls(row):
    for source_col in ["srcs_retrieved", "srcs_safe_urls", "srcs_cited"]:
        sources = row.get(source_col, [])
        if not isinstance(sources, list):
            continue
        for src in sources:
            if not isinstance(src, dict):
                continue
            url = src.get("url", "")
            if url:
                yield url


def _load_urls_content(urls_content_path=RESPONSE_URLS_CONTENT_PATH, required=True):
    if not os.path.exists(urls_content_path):
        if required:
            raise FileNotFoundError(
                f"URL content cache not found: {urls_content_path}. "
                "Run asyncio.run(extract_urls_content()) first."
            )
        return {}

    urls_content = load_json(urls_content_path)
    if urls_content is None:
        return {}
    if not isinstance(urls_content, dict):
        raise ValueError(f"Expected a JSON object at {urls_content_path}")

    return {
        str(url): content if isinstance(content, str) else ""
        for url, content in urls_content.items()
    }


async def extract_urls_content(
    urls_content_path=RESPONSE_URLS_CONTENT_PATH,
    force_refresh=False,
):
    df = _load_response_source_similarity_input()

    num_urls = 0
    unique_urls = set()
    for i, row in df.iterrows():
        row_urls = list(_iter_response_source_urls(row))
        num_urls += len(row_urls)
        unique_urls.update(row_urls)

    print(num_urls)
    print(len(unique_urls))
    print(len(df))

    url_cache = (
        {}
        if force_refresh
        else _load_urls_content(urls_content_path=urls_content_path, required=False)
    )
    checkpoint_every = max(1, URL_FETCH_CHECKPOINT_EVERY)
    processed_urls = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            for url in tqdm(sorted(unique_urls)):
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
                            "URL extraction timed out after %.1fs: %s",
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
                        except Exception as e:
                            logger.warning(
                                "Failed to relaunch browser after timeout for %s: %s",
                                url,
                                e,
                            )
                            browser = None
                    if processed_urls % checkpoint_every == 0:
                        logger.info(
                            "Checkpointing URL content cache after %s processed URLs to %s",
                            processed_urls,
                            urls_content_path,
                        )
                        to_json(url_cache, urls_content_path, indent=2)
        finally:
            if browser is not None:
                await browser.close()

    logger.info(
        "Writing final URL content cache with %s entries to %s",
        len(url_cache),
        urls_content_path,
    )
    to_json(url_cache, urls_content_path, indent=2)


def response_source_similarity(
    urls_content_path=RESPONSE_URLS_CONTENT_PATH,
):
    df = _load_response_source_similarity_input()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    num_urls = 0
    for i, row in df.iterrows():
        num_urls += len(list(_iter_response_source_urls(row)))

    print(num_urls)
    print(len(df))
    df["retrieved_sources_similarity"] = [{}] * len(df)
    df["safe_sources_similarity"] = [{}] * len(df)
    df["cited_sources_similarity"] = [{}] * len(df)
    urls_content = _load_urls_content(urls_content_path=urls_content_path)
    missing_urls = set()

    for i, row in tqdm(df.iterrows()):
        srcs_retrieved = row["srcs_retrieved"]
        srcs_safe_urls = row["srcs_safe_urls"]
        srcs_cited = row["srcs_cited"]
        asistant_response = row["asistant_response"]
        row_url_payloads = {}

        def get_similarity_payload(url):
            if url in row_url_payloads:
                return row_url_payloads[url]
            if url not in urls_content:
                missing_urls.add(url)
            content = urls_content.get(url, "")
            score = find_similarity(content, asistant_response)
            scores = scorer.score(content, asistant_response)
            nli_judge_response = compute_nli_scores(content, asistant_response)
            payload = {
                "similarity_score": score,
                "rouge_score": scores,
                "nli_judge": nli_judge_response,
                "content": content,
            }
            row_url_payloads[url] = payload
            return payload

        retrieved_urls_content = {}
        for src in srcs_retrieved:
            url = src["url"]
            retrieved_urls_content[url] = get_similarity_payload(url)

        safe_urls_content = {}
        for src in srcs_safe_urls:
            url = src["url"]
            safe_urls_content[url] = get_similarity_payload(url)

        cited_urls_content = {}
        for src in srcs_cited:
            url = src["url"]
            cited_urls_content[url] = get_similarity_payload(url)

        df.at[i, "retrieved_sources_similarity"] = retrieved_urls_content
        df.at[i, "safe_sources_similarity"] = safe_urls_content
        df.at[i, "cited_sources_similarity"] = cited_urls_content

    if missing_urls:
        logger.warning(
            "%s URLs were missing from %s and were scored with empty content",
            len(missing_urls),
            urls_content_path,
        )

    df.drop(
        columns=[
            "srcs_retrieved",
            "srcs_safe_urls",
            "srcs_cited",
            "thoughts",
            "openai_models",
            "user_msg_history",
            "interactions",
            "thinking",
        ],
        inplace=True,
    )
    df.to_csv(
        f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.csv",
        index=False,
    )
    df.to_pickle(f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.pkl")
    json_df = df.copy()
    for col in json_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        json_df[col] = json_df[col].astype(str)
    to_json(
        json_df.to_dict(orient="records"),
        f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.json",
    )

def _load_response_source_similarity_frames():
    """Build long-form source-level scores plus cited-only response-level coverage metrics."""
    df = pd.read_pickle(f"{OUTPUT_PATH}/metadata/response_and_sources_similarity.pkl")

    per_source_rows = []
    source_cols = [
        ("retrieved_sources_similarity", "Retrieved"),
        ("safe_sources_similarity", "Safe"),
        ("cited_sources_similarity", "Cited"),
    ]
    for _, row in df.iterrows():
        for source_col, source_type in source_cols:
            source_similarity = row.get(source_col, {})
            if not isinstance(source_similarity, dict):
                continue
            for url, payload in source_similarity.items():
                rouge_payload = payload.get("rouge_score", {})
                rouge_1 = 0.0
                rouge_2 = 0.0
                rouge_l = 0.0
                if isinstance(rouge_payload, dict):
                    def _rouge_precision(score_obj):
                        if hasattr(score_obj, "precision"):
                            return score_obj.precision
                        if isinstance(score_obj, dict):
                            return score_obj.get("precision", 0.0)
                        return 0.0

                    rouge_1 = _rouge_precision(rouge_payload.get("rouge1"))
                    rouge_2 = _rouge_precision(rouge_payload.get("rouge2"))
                    rouge_l = _rouge_precision(rouge_payload.get("rougeL"))
                elif hasattr(rouge_payload, "precision"):
                    rouge_l = rouge_payload.precision
                sim = payload.get("similarity_score", 0.0)
                nli_judge = payload.get("nli_judge", {}) or {}
                nli_label = str(nli_judge.get("label", "")).strip().lower()
                nli_score = int(
                    nli_judge.get("confidence", nli_judge.get("score", 0)) or 0
                )
                per_source_rows.append(
                    {
                        "user_id": row.get("user_id"),
                        "conv_id": row.get("conv_id"),
                        "turn_id": row.get("turn_id"),
                        "topic": row.get("topic"),
                        "time": pd.to_datetime(row.get("time"), errors="coerce"),
                        "url": url,
                        "source_type": source_type,
                        "response_text": row.get("asistant_response", ""),
                        "source_content": payload.get("content", ""),
                        "similarity_score": sim,
                        "rouge1_precision": rouge_1,
                        "rouge2_precision": rouge_2,
                        "rougeL_precision": rouge_l,
                        "nli_entailment": int(nli_label == "entailment") * nli_score,
                        "nli_neutral": int(nli_label == "neutral") * nli_score,
                        "nli_contradiction": int(nli_label == "contradiction") * nli_score,
                        "nli_score": nli_score,
                        "nli_label": nli_label,
                        "contradiction_reason": nli_judge.get("reasoning", ""),
                    }
                )

    per_source_df = pd.DataFrame(per_source_rows)
    if len(per_source_df) == 0:
        return per_source_df, pd.DataFrame()

    per_source_df["month"] = per_source_df["time"].dt.to_period("M").dt.to_timestamp()

    return per_source_df

def plot_response_source_quality_summary():
    df = _load_response_source_similarity_frames()
    if len(df) == 0:
        return

    row_key_cols = [
        col
        for col in ["user_id", "conv_id", "turn_id", "topic", "time", "response_text"]
        if col in df.columns
    ]

    def _make_source_types_exclusive(source_df):
        exclusive_rows = []
        for _, row_df in source_df.groupby(row_key_cols, dropna=False, sort=False):
            cited_urls = set(
                row_df.loc[row_df["source_type"] == "Cited", "url"]
                .fillna("")
                .astype(str)
            )
            safe_urls = set(
                row_df.loc[row_df["source_type"] == "Safe", "url"]
                .fillna("")
                .astype(str)
            )
            url_keys = row_df["url"].fillna("").astype(str)
            keep_mask = (
                (row_df["source_type"] == "Cited")
                | (
                    (row_df["source_type"] == "Safe")
                    & ~url_keys.isin(cited_urls)
                )
                | (
                    (row_df["source_type"] == "Retrieved")
                    & ~url_keys.isin(safe_urls | cited_urls)
                )
            )
            exclusive_rows.append(row_df.loc[keep_mask])

        if not exclusive_rows:
            return source_df.iloc[0:0].copy()
        return pd.concat(exclusive_rows, ignore_index=True)

    df = _make_source_types_exclusive(df)
    if len(df) == 0:
        return

    source_order = ["Retrieved", "Safe", "Cited"]
    color_map = {
        "Retrieved": "#636EFA",
        "Cited": "#EF553B",
        "Safe": "#00CC96",
    }

    def _plot_metric_group(
        metrics,
        file_name,
        yaxis_title,
        yaxis_range=None,
        tickformat=None,
        nli_label_filter=None,
        count_annotations=None,
    ):
        fig = go.Figure()
        for metric_col, metric_label in metrics:
            for source_type in source_order:
                subset = df[df["source_type"] == source_type]
                if nli_label_filter is not None:
                    subset = subset[subset["nli_label"] == nli_label_filter.get(metric_col, "")]
                if len(subset) == 0:
                    continue
                fig.add_trace(
                    go.Box(
                        x=[metric_label] * len(subset),
                        y=subset[metric_col],
                        name=source_type,
                        legendgroup=source_type,
                        offsetgroup=source_type,
                        marker_color=color_map[source_type],
                        boxmean=True,
                        showlegend=(metric_col == metrics[0][0]),
                    )
                )
            fig.add_vline(
                x=metric_label,
                line_width=0,
            )

        fig.update_layout(
            xaxis_title="Metric",
            yaxis_title=yaxis_title,
            boxmode="group",
        )
        if yaxis_range is not None:
            fig.update_yaxes(range=yaxis_range)
        if tickformat is not None:
            fig.update_yaxes(tickformat=tickformat)
        if count_annotations:
            for metric_col, metric_label in metrics:
                label_counts = count_annotations.get(metric_col, {})
                annotation_text = "<br>".join(
                    [
                        f"R={label_counts.get('Retrieved', 0)}",
                        f"S={label_counts.get('Safe', 0)}",
                        f"C={label_counts.get('Cited', 0)}",
                    ]
                )
                fig.add_annotation(
                    x=metric_label,
                    y=1.08,
                    xref="x",
                    yref="paper",
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    align="center",
                )
            fig.update_layout(margin=dict(t=120))
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 18))
        fig.update_xaxes(tickfont=dict(size=16))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    rouge_metrics = [
        ("rouge1_precision", "Rouge-1 Precision"),
        ("rouge2_precision", "Rouge-2 Precision"),
        ("rougeL_precision", "Rouge-L Precision"),
    ]
    _plot_metric_group(
        rouge_metrics,
        "response_source_quality_rouge_summary",
        "Rouge Precision",
        yaxis_range=[0, 1],
        tickformat=".0%",
    )

    _plot_metric_group(
        [("similarity_score", "Similarity")],
        "response_source_quality_similarity_summary",
        "Similarity",
        yaxis_range=[0, 1],
        tickformat=".0%",
    )

    def _plot_nli_label_distribution():
        nli_labels = [
            ("entailment", "Entailment"),
            ("neutral", "Neutral"),
            ("contradiction", "Contradiction"),
        ]
        valid_labels = {label for label, _display in nli_labels}
        group_keys = ["user_id", "conv_id", "turn_id", "source_type"]

        nli_rate_rows = []
        filtered_df = df[df["nli_label"].isin(valid_labels)].copy()
        for group_values, group_df in filtered_df.groupby(group_keys):
            total = len(group_df)
            if total == 0:
                continue
            row_payload = dict(zip(group_keys, group_values))
            for nli_label, label_display in nli_labels:
                count = int((group_df["nli_label"] == nli_label).sum())
                row_payload[f"{nli_label}_rate"] = count / total
            nli_rate_rows.append(row_payload)

        rate_df = pd.DataFrame(nli_rate_rows)
        if len(rate_df) == 0:
            return

        fig = go.Figure()
        for source_type in source_order:
            source_df = rate_df[rate_df["source_type"] == source_type].copy()
            if len(source_df) == 0:
                continue
            for nli_label, label_display in nli_labels:
                rate_col = f"{nli_label}_rate"
                fig.add_trace(
                    go.Box(
                        x=[label_display] * len(source_df),
                        y=source_df[rate_col],
                        name=source_type,
                        legendgroup=source_type,
                        offsetgroup=source_type,
                        marker_color=color_map[source_type],
                        boxmean=True,
                        showlegend=(nli_label == nli_labels[0][0]),
                        hovertemplate=(
                            f"{source_type}<br>{label_display}: "
                            "%{y:.1%}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            xaxis_title="NLI Label",
            yaxis_title="Rate Per Sample",
            boxmode="group",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        file_name = "response_source_quality_nli_summary"
        fig.write_html(f"{OUTPUT_PATH}/{CONF}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 18))
        fig.update_xaxes(tickfont=dict(size=16))
        fig.write_image(f"{OUTPUT_PATH}/{CONF}/{file_name}.pdf", format="pdf")

    _plot_nli_label_distribution()

    contradiction_samples = df[df["nli_label"] == "contradiction"].copy()
    for col in contradiction_samples.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        contradiction_samples[col] = contradiction_samples[col].astype(str)
    to_json(
        contradiction_samples.to_dict(orient="records"),
        f"{OUTPUT_PATH}/{CONF}/response_source_contradiction_samples.json",
    )

    valid_nli_labels = {"entailment", "neutral", "contradiction"}
    sample_with_all_labels = None
    grouped_df = df[df["nli_label"].isin(valid_nli_labels)].copy()
    for group_keys, group in grouped_df.groupby(
        ["user_id", "conv_id", "turn_id", "source_type"], dropna=False
    ):
        present_labels = set(group["nli_label"].tolist())
        if valid_nli_labels.issubset(present_labels):
            first_row = group.iloc[0]
            sample_with_all_labels = {
                "user_id": first_row.get("user_id"),
                "conv_id": first_row.get("conv_id"),
                "turn_id": first_row.get("turn_id"),
                "source_type": first_row.get("source_type"),
                "topic": first_row.get("topic"),
                "time": str(first_row.get("time")),
                "response_text": first_row.get("response_text", ""),
                "sources": [],
            }
            for _, source_row in group.iterrows():
                sample_with_all_labels["sources"].append(
                    {
                        "url": source_row.get("url", ""),
                        "nli_label": source_row.get("nli_label", ""),
                        "nli_score": source_row.get("nli_score", None),
                        "reasoning": source_row.get("contradiction_reason", ""),
                        "source_content": source_row.get("source_content", ""),
                    }
                )
            break

    if sample_with_all_labels is not None:
        def _to_json_safe(value):
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if pd.isna(value):
                return None
            return value

        sample_with_all_labels = {
            key: (
                [_to_json_safe(v) for v in value]
                if isinstance(value, list)
                else (
                    {
                        inner_key: (
                            [
                                {
                                    source_key: _to_json_safe(source_value)
                                    for source_key, source_value in source_item.items()
                                }
                                for source_item in inner_value
                            ]
                            if inner_key == "sources" and isinstance(inner_value, list)
                            else _to_json_safe(inner_value)
                        )
                        for inner_key, inner_value in value.items()
                    }
                    if isinstance(value, dict)
                    else _to_json_safe(value)
                )
            )
            for key, value in sample_with_all_labels.items()
        }
        to_json(
            sample_with_all_labels,
            f"{OUTPUT_PATH}/{CONF}/response_source_all_nli_labels_example.json",
        )


def plot_retrieved_and_cited_urls_over_time(
    output_csv_path=None,
    file_name="retrieved_and_cited_urls_over_time",
    time_freq="M",
    grounding_level="conversation",
):
    """
    Plot the average number of retrieved and cited URLs per response over time,
    along with citation rate and grounding rate.
    """
    from src.response_generation.source_selection import (
        _normalized_urls_from_sources,
        _validated_grounding_level,
    )

    grounding_level = _validated_grounding_level(grounding_level)

    def _count_urls(value):
        count = 0
        for src in _safe_parse_source_list(value):
            if not isinstance(src, dict):
                continue
            url = str(src.get("url", "") or "").strip()
            if url:
                count += 1
        return count

    def _load_response_and_sources_df_for_model(model_name):
        df = _load_response_and_sources_df()

        def _primary_openai_model(value):
            models = value
            if isinstance(value, str) and value.strip():
                try:
                    models = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    models = [value.strip()]
            if not isinstance(models, list):
                return "Unknown"
            cleaned = [model for model in models if isinstance(model, str) and model.strip()]
            if not cleaned:
                return "Unknown"
            return cleaned[-1].strip()

        df["primary_openai_model"] = df.get("openai_models", []).apply(_primary_openai_model)
        return df[df["primary_openai_model"] == model_name].copy()

    def _build_summary_df(df):
        if len(df) == 0:
            return pd.DataFrame()
        df = df.copy()
        df["time"] = pd.to_datetime(df.get("time"), errors="coerce")
        df = df[df["time"].notna()].copy()
        if len(df) == 0:
            return pd.DataFrame()

        df["retrieved_url_count"] = df["srcs_retrieved"].apply(_count_urls)
        df["cited_url_count"] = df["srcs_cited"].apply(_count_urls)

        cited_external_count_by_index = {}
        previous_retrieved_by_conv = {}
        sort_cols = [col for col in ["conv_id", "turn_id", "time"] if col in df.columns]
        iter_df = df.sort_values(sort_cols, kind="stable").copy() if sort_cols else df.copy()
        for row_index, row in iter_df.iterrows():
            retrieved_sources = row.get("srcs_retrieved", [])
            cited_sources = row.get("srcs_cited", [])
            current_retrieved_urls = set(_normalized_urls_from_sources(retrieved_sources))
            cited_urls = _normalized_urls_from_sources(cited_sources)

            if grounding_level == "conversation":
                conv_id = row.get("conv_id", None)
                previous_retrieved_urls = previous_retrieved_by_conv.get(conv_id, set())
                retrieved_urls = previous_retrieved_urls | current_retrieved_urls
                previous_retrieved_by_conv[conv_id] = retrieved_urls
            else:
                retrieved_urls = current_retrieved_urls

            cited_external_count_by_index[row_index] = sum(
                1 for url in cited_urls if url in retrieved_urls
            )

        df["cited_retrieved_count"] = df.index.map(
            lambda idx: cited_external_count_by_index.get(idx, 0)
        )
        df["citation_rate"] = np.where(
            df["retrieved_url_count"] > 0,
            df["cited_retrieved_count"] / df["retrieved_url_count"],
            0.0,
        )
        df["grounding_rate"] = np.where(
            df["cited_url_count"] > 0,
            df["cited_retrieved_count"] / df["cited_url_count"],
            0.0,
        )
        df["time_period"] = df["time"].dt.to_period(time_freq).dt.to_timestamp()

        return (
            df.groupby("time_period", dropna=False)
            .agg(
                response_count=("conv_id", "size"),
                avg_retrieved_url_count=("retrieved_url_count", "mean"),
                avg_cited_url_count=("cited_url_count", "mean"),
                avg_cited_retrieved_count=("cited_retrieved_count", "mean"),
                total_retrieved_url_count=("retrieved_url_count", "sum"),
                total_cited_url_count=("cited_url_count", "sum"),
                total_cited_retrieved_count=("cited_retrieved_count", "sum"),
                citation_rate=("citation_rate", "mean"),
                grounding_rate=("grounding_rate", "mean"),
            )
            .reset_index()
            .sort_values("time_period", kind="stable")
        )

    def _write_plot(summary_df, destination_dir, output_stem):
        if len(summary_df) == 0:
            return
        os.makedirs(destination_dir, exist_ok=True)
        summary_df.to_csv(f"{destination_dir}/{output_stem}.csv", index=False)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=summary_df["time_period"],
                y=summary_df["avg_retrieved_url_count"],
                mode="lines+markers",
                name="Search Result URLs",
                line=dict(color="#636EFA", width=3),
                marker=dict(size=8),
                customdata=np.column_stack(
                    [
                        summary_df["total_retrieved_url_count"],
                        summary_df["response_count"],
                    ]
                ),
                hovertemplate=(
                    "Time: %{x|%Y-%m}<br>"
                    "Avg retrieved URLs: %{y:.2f}<br>"
                    "Total retrieved URLs: %{customdata[0]}<br>"
                    "Responses: %{customdata[1]}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=summary_df["time_period"],
                y=summary_df["avg_cited_url_count"],
                mode="lines+markers",
                name="Cited URLs",
                line=dict(color="#EF553B", width=3),
                marker=dict(size=8),
                customdata=np.column_stack(
                    [
                        summary_df["total_cited_url_count"],
                        summary_df["response_count"],
                    ]
                ),
                hovertemplate=(
                    "Time: %{x|%Y-%m}<br>"
                    "Avg cited URLs: %{y:.2f}<br>"
                    "Total cited URLs: %{customdata[0]}<br>"
                    "Responses: %{customdata[1]}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=summary_df["time_period"],
                y=summary_df["citation_rate"],
                mode="lines+markers",
                name="Citation Rate",
                line=dict(color="#00CC96", width=3),
                marker=dict(size=8),
                hovertemplate=(
                    "Time: %{x|%Y-%m}<br>"
                    "Citation rate: %{y:.2%}<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=summary_df["time_period"],
                y=summary_df["grounding_rate"],
                mode="lines+markers",
                name="Grounding Rate",
                line=dict(color="#AB63FA", width=3),
                marker=dict(size=8),
                hovertemplate=(
                    "Time: %{x|%Y-%m}<br>"
                    "Grounding rate: %{y:.2%}<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            xaxis_title="Time",
        )
        fig.update_yaxes(title_text="Average URLs Per Response", secondary_y=False)
        fig.update_yaxes(title_text="Rate", tickformat=".0%", secondary_y=True)
        if str(time_freq).upper() == "M":
            fig.update_xaxes(dtick="M1", tickangle=-45)
        else:
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=18))

        fig.write_html(f"{destination_dir}/{output_stem}.html")
        try:
            paper_fig = with_paper_style(fig, config=styler(22, 22), legend_pos=(0.9, 1.3))
            paper_fig.update_xaxes(tickfont=dict(size=18))
            paper_fig.write_image(f"{destination_dir}/{output_stem}.pdf", format="pdf")
        except Exception as e:
            logger.warning("Could not write retrieved/cited URL over-time PDF: %s", e)

    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)

    summary_df = _build_summary_df(_load_response_and_sources_df())
    if len(summary_df) == 0:
        return summary_df

    if output_csv_path is not None:
        summary_df.to_csv(output_csv_path, index=False)
    _write_plot(summary_df, output_dir, file_name)

    per_model_dir = os.path.join(output_dir, f"{file_name}_by_openai_model")
    all_model_frames = []
    model_source_df = _load_response_and_sources_df()
    if "openai_models" in model_source_df.columns:
        def _primary_openai_model(value):
            models = value
            if isinstance(value, str) and value.strip():
                try:
                    models = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    models = [value.strip()]
            if not isinstance(models, list):
                return "Unknown"
            cleaned = [model for model in models if isinstance(model, str) and model.strip()]
            if not cleaned:
                return "Unknown"
            return cleaned[-1].strip()
        model_source_df["primary_openai_model"] = model_source_df["openai_models"].apply(
            _primary_openai_model
        )
        model_names = [
            model_name
            for model_name in model_source_df["primary_openai_model"].dropna().astype(str).unique().tolist()
            if model_name
        ]
    else:
        model_names = []

    for model_name in sorted(model_names):
        model_summary_df = _build_summary_df(
            _load_response_and_sources_df_for_model(model_name)
        )
        if len(model_summary_df) == 0:
            continue
        model_summary_df = model_summary_df.copy()
        model_summary_df["model"] = model_name
        all_model_frames.append(model_summary_df)
        _write_plot(
            model_summary_df,
            per_model_dir,
            f"{file_name}_{re.sub(r'[^A-Za-z0-9._-]+', '_', model_name)}",
        )

    if all_model_frames:
        pd.concat(all_model_frames, ignore_index=True, sort=False).to_csv(
            os.path.join(per_model_dir, f"{file_name}_all_models.csv"),
            index=False,
        )

    return summary_df


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
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")
    chunking_method = _normalize_chunking_method(chunking_method)
    claim_selection_mode = _normalize_claim_selection_mode(claim_selection_mode)
    source_text_mode = _normalize_source_text_mode(source_text_mode)
    method_base = (
        RESPONSE_SOURCE_NLI_SENTENCE_BASED_BERT_BASE
        if nli_method == "bert"
        else RESPONSE_SOURCE_NLI_SENTENCE_BASED_JUDGE_BASE
    )
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
    claim_cache_path=CLAIM_EXTRACTION_CACHE_PATH,
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
    )
    persisted_claims_cache = _load_claims_cache(cache_path=claim_cache_path)
    claims_cache_dirty = False
    new_claim_cache_entries = 0

    df = _load_response_source_similarity_input()
    urls_content_by_clean_url = {}

    urls_content = _load_urls_content(
        urls_content_path=RESPONSE_URLS_CONTENT_PATH,
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
    user_query_lookup = _build_response_source_user_query_lookup()
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
    cache_path=CLAIM_EXTRACTION_CACHE_PATH,
    model_name=FACTUALITY_JUDGE_MODEL,
    output_suffix="_factuality",
):
    """
    Load the cached extracted claims JSON, evaluate each claim's factuality,
    and save an enriched cache-shaped JSON file alongside the original cache.
    """
    claims_cache = _load_claims_cache(cache_path=cache_path)
    if not claims_cache:
        raise FileNotFoundError(
            f"Claim cache not found or empty at {cache_path}."
        )

    cache_key_to_user_query = _build_claim_cache_user_query_lookup()
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
    input_path=f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge_claim.json",
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
    input_path=f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge_claim.json",
    tranco_input_path=None,
    file_name="claim_bucket_tranco_rank_comparison",
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

    claim_records = load_json(input_path)
    if not isinstance(claim_records, list):
        raise ValueError(f"Expected a JSON list at {input_path}")

    tranco_input_path = tranco_input_path or (
        f"{OUTPUT_PATH}/metadata/response_and_sources_with_tranco_ranks.pkl"
    )
    if not os.path.exists(tranco_input_path):
        raise FileNotFoundError(
            f"Tranco-ranked source metadata not found at {tranco_input_path}. "
            "Run the Tranco-ranking pipeline first."
        )

    # Reuse the existing Tranco evaluation helper so the standard rank artifacts
    # and sanity checks stay in sync with this analysis.
    try:
        evaluate_source_tranco_ranks()
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

    output_dir = f"{OUTPUT_PATH}/{CONF}"
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
    input_path=f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge_claim_snippet.json",
):
    """
    Diagnose why snippet-mode judge outputs produce many Unexplained rows.
    Prints bucket counts plus snippet-coverage stats for checked sources.
    """
    source_records = load_json(input_path)
    if not isinstance(source_records, list):
        raise ValueError(f"Expected a JSON list at {input_path}")

    response_df = _load_response_source_similarity_input().copy()
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
    full_input_path=f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge_claim.json",
    snippet_input_path=f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge_claim_snippet.json",
    sample_size=10,
    random_state=42,
    output_path=None,
):
    """
    Randomly sample aligned response chunks from full-text and snippet runs and
    save a side-by-side comparison with each method's selected URL, full
    content, snippet metadata, entailment-judge output, and final bucket.
    """
    full_records = load_json(full_input_path)
    snippet_records = load_json(snippet_input_path)
    if not isinstance(full_records, list):
        raise ValueError(f"Expected a JSON list at {full_input_path}")
    if not isinstance(snippet_records, list):
        raise ValueError(f"Expected a JSON list at {snippet_input_path}")

    if output_path is None:
        output_path = (
            f"{OUTPUT_PATH}/metadata/"
            "response_source_nli_method_comparison_samples.json"
        )

    response_df = _load_response_source_similarity_input().copy()
    urls_content = _load_urls_content(
        urls_content_path=RESPONSE_URLS_CONTENT_PATH,
        required=False,
    )
    user_query_lookup = _build_response_source_user_query_lookup()

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
):
    if nli_method not in {"bert", "judge"}:
        raise ValueError("nli_method must be one of {'bert', 'judge'}")

    chunking_method = _normalize_chunking_method(chunking_method)
    output_base = output_base or _response_source_nli_output_base(
        nli_method,
        chunking_method,
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

    output_dir = f"{OUTPUT_PATH}/{CONF}"
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


def plot_response_source_nli_entailment_score_boxplots_all():
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
    output_dir = f"{OUTPUT_PATH}/{CONF}"
    os.makedirs(output_dir, exist_ok=True)
    combined_summary_df.to_csv(
        f"{output_dir}/response_source_nli_entailment_score_boxplot_all_summary.csv",
        index=False,
    )
    return combined_summary_df


if __name__ == "__main__":
    web_df = load_web_data_from_file(fmt="pkl")
    print(f"Loaded web data: {len(web_df)}")
    extract_response_and_sources(web_df)

    asyncio.run(extract_urls_content(force_refresh=True))
    # plot_response_source_quality_summary()

    # response_source_nli_sentence_based(nli_method="judge", chunking_method="claim", claim_selection_mode="all", source_text_mode="snippet")
    # plot_response_source_nli_sentence_based_judge(
    #     chunking_method="claim",
    #     source_text_mode="snippet",
    #     claim_selection_mode="all", 
    # )
    # plot_response_source_nli_sentence_based_judge(
    #     chunking_method="claim",
    #     # source_text_mode="snippet",
    #     claim_selection_mode="all", 
    # )

    # plot_response_source_nli_entailment_score_boxplot(nli_method="judge", chunking_method="claim")

    # response_source_nli_sentence_based_factuality(
    #     input_path="outputs/metadata/response_source_nli_sentence_based_judge_claim.json"
    # )

    # summarize_response_source_nli_sentence_based_factuality(
    #     # input_path=f"{OUTPUT_PATH}/metadata/response_source_nli_sentence_based_judge_claim.json"
    # )

    # sample_response_source_nli_method_comparison()

    # plot_retrieved_and_cited_urls_over_time()

    # plot_claim_bucket_tranco_rank_comparison()

    # bootstrap_response_source_nli_sentence_based_confidence_intervals(
    #     nli_method="judge",
    #     chunking_method="claim",
    #     claim_selection_mode="all",
    #     source_text_mode="full_url_content",
    #     n_boot=1000,
    #     random_state=42,
    # )
    pass
