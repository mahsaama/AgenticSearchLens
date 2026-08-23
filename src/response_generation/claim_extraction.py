"""Extracts atomic factual claims from an LLM response's text, with a
content-hash cache so the same response text is never re-sent to the
extraction model twice.

extract_claims_from_text(text) is the entry point: one LLM call
(CLAIM_EXTRACTION_MODEL) against SYSTEM_PROMPT_CLAIM_EXTRACTION, parsed and
cleaned into a flat list of claim strings. The rest of this module is the
claim cache built around it -- entailment_analysis.py's
response_source_nli_sentence_based() looks claims up by a hash of the
response-chunk text before calling extract_claims_from_text() itself, and
response_source_claim_cache_factuality() replays factuality scoring over
whatever's already in the cache.

Split out of response_generation.py, alongside web_content_fetch.py and
entailment_analysis.py -- response_generation.py imports the entry points
it needs from all three rather than defining them itself.

Run directly (`python -m src.response_generation.claim_extraction`) for a
small smoke test: extracts claims from one hardcoded example paragraph and
prints them (needs OPENAI_API_KEY; doesn't touch outputs/ or the cache
file).
"""

import ast
import hashlib
import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.prompts.evaluator_prompts import (
    SYSTEM_PROMPT_CLAIM_EXTRACTION,
    USER_PROMPT_CLAIM_EXTRACTION,
)
from src.utils.common_io import load_json, to_json
from src.response_generation.web_content_fetch import _load_response_source_similarity_input

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLAIM_EXTRACTION_MODEL = os.getenv("CLAIM_EXTRACTION_MODEL")
CLAIM_EXTRACTION_MAX_INPUT_CHARS = int(os.getenv("CLAIM_EXTRACTION_MAX_INPUT_CHARS"))
CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS = int(
    os.getenv("CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS")
)
CLAIM_EXTRACTION_CACHE_PATH = os.getenv("CLAIM_EXTRACTION_CACHE_PATH")

def extract_first_json_object(text):
    """Parse `text` as JSON outright, or -- despite the name -- find the
    first balanced {...} object in it via brace-depth counting (not a
    naive first-`{`-to-last-`}` slice, which breaks when a provider's
    response text concatenates more than one JSON blob together: seen in
    practice from xAI's Responses API under forced multi-round tool use,
    where output_text can hold several {...} fragments plus stray
    continuation tokens back to back)."""
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return ast.literal_eval(candidate)
                        except (ValueError, SyntaxError):
                            pass
                    break
        start = text.find("{", start + 1)
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


def _build_claim_cache_user_query_lookup(platform="chatgpt"):
    response_df = _load_response_source_similarity_input(platform=platform).copy()
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

def _smoke_test():
    """Standalone sanity check: extract claims from one hardcoded example
    paragraph and print them. Needs OPENAI_API_KEY; doesn't touch the
    on-disk claim cache or outputs/.
    """
    sample_text = (
        "The Eiffel Tower was completed in 1889 and stands 330 meters tall. "
        "It was designed by the engineer Gustave Eiffel for the 1889 World's Fair."
    )
    claims = extract_claims_from_text(sample_text)
    print(f"Extracted {len(claims)} claim(s):")
    for claim in claims:
        print(f"  - {claim}")


if __name__ == "__main__":
    _smoke_test()
