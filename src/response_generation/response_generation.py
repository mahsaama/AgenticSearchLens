"""§5.2 analyses: how responses are grounded in (or ungrounded from) their
cited/retrieved sources -- scraping cited URLs' actual content, computing
NLI entailment between response claims and that content, and factuality
scoring by grounding source (associated citation / other citation / search
result / parametric knowledge).

The three heaviest pieces of this pipeline each live in their own module
now -- this file is the orchestrator that imports and runs them, plus
whatever doesn't belong to any of the three (raw response_and_sources
extraction, the embedding-similarity grounding check, and the plots built
directly on it):

- src/response_generation/web_content_fetch.py: fetches and caches the raw
  text of cited/retrieved source URLs.
- src/response_generation/claim_extraction.py: extracts atomic claims from
  a response's text, with a content-hash cache.
- src/response_generation/entailment_analysis.py: NLI entailment scoring
  and the factuality/grounding-source analysis built on it -- the bulk of
  §5.2's figures and tables.

Same scope note as query_reformulations.py / source_selection.py: written
for the paper's full cohort, organized as a library of individually-
runnable analysis functions (see the __main__ call list), each writing its
own figure/table under outputs/response_generation/.

Pipeline dependency: extract_response_and_sources(web_df) (and
extract_response_and_sources_other_platforms() for non-ChatGPT platforms)
writes outputs/<platform>/metadata/response_and_sources.pkl -- most of
the grounding/NLI functions here read it, and it's also the prerequisite
source_selection.py's count_unique_retrieved_safe_cited() and related
functions need but don't produce themselves. Run it before those.
"""

import os
import json
import re
import ast
import logging
import asyncio
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

pio.defaults.mathjax = None
from src.utils.common_io import *
from src.utils.chatgpt_conversation_utils import *
from src.utils.figure_style import with_paper_style, styler
from src.web_search_decision.extraction import load_web_data_from_file

# Module import (not just the two names actually called below) so every
# other entailment_analysis.py function -- response_source_nli_sentence_
# based(), the plotting/bootstrapping functions, etc. -- is reachable as
# ea.<name> from __main__ without a growing per-function import list.
import src.response_generation.entailment_analysis as ea
from src.response_generation.entailment_analysis import compute_nli_scores, _safe_parse_source_list
from src.response_generation.web_content_fetch import (
    RESPONSE_URLS_CONTENT_PATH,
    _iter_response_source_urls,
    _load_response_source_similarity_input,
    _load_urls_content,
    extract_urls_content,
)

CONF = "./response_generation"


def _platform_metadata_dir(platform):
    """outputs/<platform>/metadata for every platform, ChatGPT included --
    same convention as extraction.py's _metadata_dir(). Used by
    extract_response_and_sources[_other_platforms] so Claude/Grok/DeepSeek
    runs don't overwrite ChatGPT's response_and_sources.pkl (and each
    other's) by all writing to the same path."""
    return f"{OUTPUT_PATH}/{platform}/metadata"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def find_similarity(page_content, response):
    embeddings = model.encode([response, page_content])

    response_emb = embeddings[0]
    page_content_embs = embeddings[1:]

    scores = cosine_similarity(
        response_emb.reshape(1, -1), page_content_embs.reshape(1, -1)
    )[0]

    return float(scores.mean())





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



def _load_response_and_sources_df(platform="chatgpt"):
    pkl_path = f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources.pkl"
    csv_path = f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources.csv"

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


def response_source_similarity(
    urls_content_path=None,
    platform="chatgpt",
):
    if urls_content_path is None:
        urls_content_path = (
            RESPONSE_URLS_CONTENT_PATH
            if platform == "chatgpt"
            else f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_url_content.json"
        )
    df = _load_response_source_similarity_input(platform=platform)

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
        f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_similarity.csv",
        index=False,
    )
    df.to_pickle(f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_similarity.pkl")
    json_df = df.copy()
    for col in json_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        json_df[col] = json_df[col].astype(str)
    to_json(
        json_df.to_dict(orient="records"),
        f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_similarity.json",
    )

def _load_response_source_similarity_frames(platform="chatgpt"):
    """Build long-form source-level scores plus cited-only response-level coverage metrics."""
    df = pd.read_pickle(f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_similarity.pkl")

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
        return per_source_df

    per_source_df["month"] = per_source_df["time"].dt.to_period("M").dt.to_timestamp()

    return per_source_df

def plot_response_source_quality_summary(platform="chatgpt"):
    df = _load_response_source_similarity_frames(platform=platform)
    if len(df) == 0:
        return
    output_dir = f"{OUTPUT_PATH}/{CONF}/{platform}"
    os.makedirs(output_dir, exist_ok=True)

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
        fig.write_html(f"{output_dir}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 18))
        fig.update_xaxes(tickfont=dict(size=16))
        fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

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
        fig.write_html(f"{output_dir}/{file_name}.html")
        fig = with_paper_style(fig, config=styler(18, 18))
        fig.update_xaxes(tickfont=dict(size=16))
        fig.write_image(f"{output_dir}/{file_name}.pdf", format="pdf")

    _plot_nli_label_distribution()

    contradiction_samples = df[df["nli_label"] == "contradiction"].copy()
    for col in contradiction_samples.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        contradiction_samples[col] = contradiction_samples[col].astype(str)
    to_json(
        contradiction_samples.to_dict(orient="records"),
        f"{output_dir}/response_source_contradiction_samples.json",
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
            f"{output_dir}/response_source_all_nli_labels_example.json",
        )


def plot_retrieved_and_cited_urls_over_time(
    output_csv_path=None,
    file_name="retrieved_and_cited_urls_over_time",
    time_freq="M",
    grounding_level="conversation",
    platform="chatgpt",
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
        df = _load_response_and_sources_df(platform=platform)

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

    output_dir = f"{OUTPUT_PATH}/{CONF}/{platform}"
    os.makedirs(output_dir, exist_ok=True)

    summary_df = _build_summary_df(_load_response_and_sources_df(platform=platform))
    if len(summary_df) == 0:
        return summary_df

    if output_csv_path is not None:
        summary_df.to_csv(output_csv_path, index=False)
    _write_plot(summary_df, output_dir, file_name)

    per_model_dir = os.path.join(output_dir, f"{file_name}_by_openai_model")
    all_model_frames = []
    model_source_df = _load_response_and_sources_df(platform=platform)
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



if __name__ == "__main__":
    # Run the extraction stage for every platform we have extracted
    # data_summary/web_data_summary for (see src.utils.common_io.PLATFORMS).
    for platform in PLATFORMS:
        try:
            web_df = load_web_data_from_file(fmt="pkl", platform=platform)
        except Exception as e:
            print(f"[{platform}] Skipping -- failed to load web data: {e}")
            continue
        print(f"[{platform}] Loaded web data: {len(web_df)}")

        if platform == "chatgpt":
            extract_response_and_sources(web_df)
        else:
            extract_response_and_sources_other_platforms(web_df, platform)

        asyncio.run(extract_urls_content(force_refresh=True, platform=platform))
        # plot_response_source_quality_summary(platform=platform)

    # ea.response_source_nli_sentence_based(nli_method="judge", chunking_method="claim", claim_selection_mode="all", source_text_mode="snippet")
    # ea.plot_response_source_nli_sentence_based_judge(
    #     chunking_method="claim",
    #     source_text_mode="snippet",
    #     claim_selection_mode="all",
    # )
    # ea.plot_response_source_nli_sentence_based_judge(
    #     chunking_method="claim",
    #     # source_text_mode="snippet",
    #     claim_selection_mode="all",
    # )

    # ea.plot_response_source_nli_entailment_score_boxplot(nli_method="judge", chunking_method="claim")

    # ea.response_source_nli_sentence_based_factuality(
    #     input_path="outputs/chatgpt/metadata/response_source_nli_sentence_based_judge_claim.json"
    # )

    # ea.summarize_response_source_nli_sentence_based_factuality(
    #     # input_path=f"{OUTPUT_PATH}/chatgpt/metadata/response_source_nli_sentence_based_judge_claim.json"
    # )

    # ea.sample_response_source_nli_method_comparison()

    # plot_retrieved_and_cited_urls_over_time()

    # ea.plot_claim_bucket_tranco_rank_comparison()

    # ea.bootstrap_response_source_nli_sentence_based_confidence_intervals(
    #     nli_method="judge",
    #     chunking_method="claim",
    #     claim_selection_mode="all",
    #     source_text_mode="full_url_content",
    #     n_boot=1000,
    #     random_state=42,
    # )
    pass
