"""
End-to-end hallucinated URL flow from a response_and_sources.pkl file.

Input pkl requirements:
  - srcs_retrieved: retrieved source objects/URLs for each response
  - srcs_cited: cited source objects/URLs for each response
  - conv_id: conversation identifier
  - optional turn_id/time/month: used for stable ordering and Wayback timestamp

Outputs under --output-dir:
  - cited_and_retrieved_occurrences.json
  - cited_only_occurrences.json
  - cited_and_retrieved_unique.json
  - cited_only_unique.json
  - cited_and_retrieved_cite_months.json
  - cited_only_cite_months.json
  - cited_and_retrieved_reachability.json
  - cited_only_reachability.json
  - cited_and_retrieved_wayback.json
  - cited_only_wayback.json
  - cited_and_retrieved_classified.json
  - cited_only_classified.json
  - cited_and_retrieved_hallucinated.json
  - cited_only_hallucinated.json
  - results_hallucinated_rate.json

Example:
  python -m src.response_generation.hallucinated_url_detection \
    --input-pkl /path/to/response_and_sources.pkl \
    --output-dir hallucinated_url_results
"""

import argparse
import asyncio
import ast
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunparse, urlunsplit


CONCURRENCY = 20
TIMEOUT = 20
MAX_RETRIES = 3
DEAD_CODES = {404, 410}
WAYBACK_API = "https://archive.org/wayback/available"
RATE_LIMIT_SLEEP = 5.0
MAX_CONSECUTIVE_429 = 2
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class RateLimited(Exception):
    """Raised internally when the Wayback Machine API returns HTTP 429, so
    run_wayback_group can back off instead of treating it as a normal error."""
    pass


def normalize_url(url):
    """Canonicalize a URL for deduplication: strip a trailing "/" from the
    path and drop a chatgpt/openai utm_source tracking param, so the same
    page cited with/without tracking params or a trailing slash counts once."""
    if not isinstance(url, str):
        return ""
    value = url.strip()
    if not value:
        return ""

    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value.rstrip("/")

    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not (key == "utm_source" and val in {"chatgpt.com", "openai"})
    ]
    clean_path = parts.path.rstrip("/") if parts.path != "/" else ""
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            clean_path,
            urlencode(query, doseq=True),
            parts.fragment,
        )
    )


def parse_source_value(value):
    """Coerce a pkl column's raw source value -- which may already be a
    list/tuple/set, a stringified-list (as stored after a CSV/pickle
    round-trip), NaN, or None -- into an actual Python list."""
    if value is None:
        return []
    try:
        import pandas as pd

        if pd.isna(value):
            return []
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return parse_source_value(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return [value]
    return []


def source_urls(value):
    """Extract every normalized http(s) URL from a source-column value:
    recurses into nested lists, and pulls the URL out of dicts under any of
    "url"/"source_url"/"link"/"uri"."""
    urls = []

    def collect(item):
        if isinstance(item, dict):
            url = (
                item.get("url")
                or item.get("source_url")
                or item.get("link")
                or item.get("uri")
            )
            if url:
                clean = normalize_url(url)
                if clean:
                    urls.append(clean)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                collect(nested)
        elif isinstance(item, str):
            clean = normalize_url(item)
            if clean.startswith(("http://", "https://")):
                urls.append(clean)

    collect(parse_source_value(value))
    return urls


def dump_json(path, data):
    """Write `data` as pretty-printed JSON to `path`, creating parent
    directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_json(path, default):
    """Load JSON from `path`, or return `default` if it doesn't exist yet
    -- every stage of this pipeline uses this to resume from wherever a
    previous run left off."""
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def row_month(row):
    """First-of-month string (YYYY-MM-01) for a response row's "month" or
    "time" field, or None if neither is present/parseable. Used as the
    Wayback Machine lookup timestamp, so a citation is checked against a
    snapshot close to when it was actually cited."""
    for key in ("month", "time"):
        value = row.get(key)
        if value is None:
            continue
        try:
            if hasattr(value, "strftime"):
                return value.strftime("%Y-%m-01")
            text = str(value)
            if len(text) >= 7:
                return f"{text[:7]}-01"
        except (TypeError, ValueError):
            continue
    return None


def iter_grounding_rows(df):
    """Yield (index, row) pairs from `df` in stable conv_id/turn_id/time
    order (whichever of those columns are present), so citation-group
    bucketing is deterministic across runs."""
    sort_cols = [col for col in ("conv_id", "turn_id", "time") if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")
    return df.iterrows()


def build_citation_groups(input_pkl, output_dir):
    """Stage 1: split every cited URL in `input_pkl` into "cited_and_retrieved"
    (also appeared in that conversation's own retrieved sources, cumulative
    across turns) vs. "cited_only" (cited but never actually retrieved --
    the group most likely to contain hallucinated citations), writing
    per-group occurrences/unique-URLs/cite-months JSON under `output_dir`.
    Returns a summary dict (also written to citation_grouping_summary.json).
    """
    with Path(input_pkl).open("rb") as f:
        df = pickle.load(f)

    conversation_retrieved = defaultdict(set)
    occurrences = {
        "cited_and_retrieved": [],
        "cited_only": [],
    }
    cite_months = {
        "cited_and_retrieved": {},
        "cited_only": {},
    }

    num_responses = len(df)
    retrieval_count = 0
    citation_count = 0

    for row_index, row in iter_grounding_rows(df):
        retrieved_urls = source_urls(row.get("srcs_retrieved"))
        cited_urls = source_urls(row.get("srcs_cited"))
        retrieval_count += len(retrieved_urls)
        citation_count += len(cited_urls)

        conv_id = row.get("conv_id")
        retrieved_set = conversation_retrieved[conv_id]
        retrieved_set.update(retrieved_urls)
        month = row_month(row)

        for url in cited_urls:
            group = "cited_and_retrieved" if url in retrieved_set else "cited_only"
            occurrence = {
                "url": url,
                "group": group,
                "conv_id": None if conv_id is None else str(conv_id),
                "turn_id": None if row.get("turn_id") is None else str(row.get("turn_id")),
                "row_index": None if row_index is None else str(row_index),
                "cite_month": month,
            }
            occurrences[group].append(occurrence)
            cite_months[group].setdefault(url, month)

    summary = {
        "input_pkl": str(input_pkl),
        "counting_basis": "occurrence_weighted_conversation_cumulative_cited_urls",
        "num_responses": int(num_responses),
        "sum_retrievals": int(retrieval_count),
        "sum_citations": int(citation_count),
        "groups": {},
    }

    for group, rows in occurrences.items():
        unique_urls = sorted({row["url"] for row in rows})
        dump_json(output_dir / f"{group}_occurrences.json", rows)
        dump_json(output_dir / f"{group}_unique.json", unique_urls)
        dump_json(output_dir / f"{group}_cite_months.json", cite_months[group])
        summary["groups"][group] = {
            "occurrences": len(rows),
            "unique_urls": len(unique_urls),
        }

    dump_json(output_dir / "citation_grouping_summary.json", summary)
    return summary


async def check_url(session, url, semaphore):
    """Check whether `url` is reachable (HEAD, falling back to GET on
    405/406), retrying transient failures up to MAX_RETRIES times. Returns
    the HTTP status code on success, or a short string label describing why
    it couldn't be checked (e.g. "TIMEOUT", "CONNECT_ERROR")."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await session.head(url, allow_redirects=True, timeout=TIMEOUT)
                if response.status_code in (405, 406):
                    response = await session.get(
                        url, allow_redirects=True, timeout=TIMEOUT
                    )
                return response.status_code
            except Exception as exc:
                err = str(exc).lower()
                if "timeout" in err:
                    label = "TIMEOUT"
                elif "connect" in err or "resolve" in err or "dns" in err:
                    label = "CONNECT_ERROR"
                elif "redirect" in err:
                    return "TOO_MANY_REDIRECTS"
                else:
                    label = f"ERROR:{type(exc).__name__}"
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                return label
    return "UNKNOWN"


async def run_reachability_group(output_dir, group):
    """Stage 2: check every not-yet-checked URL in `{group}_unique.json`
    concurrently (CONCURRENCY at a time, browser-impersonating client to
    reduce bot-blocking) and write the results to `{group}_reachability.json`,
    checkpointing every 100 URLs so a long run can be resumed."""
    from curl_cffi.requests import AsyncSession

    urls = load_json(output_dir / f"{group}_unique.json", [])
    out_path = output_dir / f"{group}_reachability.json"
    status = load_json(out_path, {})
    to_check = [url for url in urls if url not in status]
    print(f"[{group}] reachability {len(to_check):,}/{len(urls):,} URLs")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with AsyncSession(impersonate="chrome") as session:
        tasks = {
            url: asyncio.create_task(check_url(session, url, semaphore))
            for url in to_check
        }
        for done, (url, task) in enumerate(tasks.items(), start=1):
            status[url] = await task
            if done % 100 == 0 or done == len(tasks):
                dump_json(out_path, status)

    dump_json(out_path, status)
    return status


def check_wayback(client, url, timestamp):
    """Query the Wayback Machine's "available" API for the closest archived
    snapshot of `url` at/near `timestamp` (YYYYMMDD, or ""). Returns
    (has_snapshot, snapshot_dict); raises RateLimited on HTTP 429."""
    query = f"{WAYBACK_API}?url={url}"
    if timestamp:
        query += f"&timestamp={timestamp}"
    response = client.get(query)
    if response.status_code == 429:
        raise RateLimited("HTTP 429")
    data = response.json()
    snapshots = data.get("archived_snapshots", {})
    if snapshots:
        return True, snapshots["closest"]
    return False, None


def is_wikipedia_url(url):
    """True if `url`'s host is wikipedia.org or a wikipedia.org subdomain
    (e.g. en.wikipedia.org)."""
    host = urlparse(url).hostname or ""
    host = host.lower()
    return host == "wikipedia.org" or host.endswith(".wikipedia.org")


def wikipedia_history_url(url):
    """`url` rewritten to its "?action=history" edit-history page."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["action"] = "history"
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def check_wikipedia_history(client, url):
    """For a dead (404/410) Wikipedia URL, check its edit-history page for
    evidence the article existed and was later deleted/moved ("stale") vs.
    no such evidence ("unknown") -- a Wikipedia-specific fallback for pages
    the Wayback Machine hasn't archived. Returns (looks_stale, details_dict);
    (False, {}) immediately if `url` isn't a Wikipedia URL."""
    if not is_wikipedia_url(url):
        return False, {}

    history_url = wikipedia_history_url(url)
    response = client.get(history_url, follow_redirects=True)
    body = response.text.lower()
    details = {
        "wikipedia_history_url": history_url,
        "wikipedia_history_status": response.status_code,
    }
    if not 200 <= response.status_code < 400:
        details["wikipedia_history_class"] = "not_found"
        return False, details

    no_history_markers = (
        "there is no edit history for this page",
        "this page does not have a history",
        "no revisions were found",
    )
    if any(marker in body for marker in no_history_markers):
        details["wikipedia_history_class"] = "not_found"
        return False, details

    history_markers = ("id=\"pagehistory\"", "mw-history", "mw-changeslist", "deletion log")
    if any(marker in body for marker in history_markers):
        details["wikipedia_history_class"] = "stale"
        return True, details

    details["wikipedia_history_class"] = "unknown"
    return False, details


def run_wayback_group(output_dir, group):
    """Stage 3: for every URL in `{group}` that came back dead (404/410,
    from stage 2's reachability check) and hasn't been checked yet, look it
    up in the Wayback Machine (falling back to a Wikipedia edit-history
    check) and classify it "stale" (existed, later removed) or
    "hallucinated" (no evidence it ever existed). Backs off on repeated
    HTTP 429s from the Wayback API and checkpoints every 25 URLs. Writes/
    returns `{group}_wayback.json`."""
    import httpx

    reach = load_json(output_dir / f"{group}_reachability.json", {})
    months = load_json(output_dir / f"{group}_cite_months.json", {})
    out_path = output_dir / f"{group}_wayback.json"
    results = load_json(out_path, {})
    targets = []
    for url, status in reach.items():
        try:
            code = int(status)
        except (TypeError, ValueError):
            continue
        if code in DEAD_CODES and url not in results:
            targets.append((url, code, months.get(url)))

    print(f"[{group}] wayback {len(targets):,} missing 404/410 URLs")
    consecutive_429 = 0
    with httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        for i, (url, status, month) in enumerate(targets, start=1):
            timestamp = month.replace("-", "")[:8] if month else ""
            try:
                has_snapshot, snapshot = check_wayback(client, url, timestamp)
                consecutive_429 = 0
                if has_snapshot:
                    results[url] = {
                        "status": status,
                        "cite_month": month,
                        "wayback_class": "stale",
                        "wayback_snapshot": snapshot.get("url"),
                        "wayback_timestamp": snapshot.get("timestamp"),
                    }
                else:
                    wiki_has_history, wiki_details = check_wikipedia_history(client, url)
                    results[url] = {
                        "status": status,
                        "cite_month": month,
                        "wayback_class": "stale" if wiki_has_history else "hallucinated",
                        "wayback_snapshot": None,
                        "wayback_timestamp": None,
                        **wiki_details,
                    }
            except RateLimited:
                consecutive_429 += 1
                dump_json(out_path, results)
                if consecutive_429 >= MAX_CONSECUTIVE_429:
                    print(f"[{group}] stopping after {consecutive_429} consecutive 429s")
                    break
                time.sleep(RATE_LIMIT_SLEEP)
                continue
            except Exception as exc:
                consecutive_429 = 0
                results[url] = {
                    "status": status,
                    "cite_month": month,
                    "wayback_class": "error",
                    "wayback_error": f"{type(exc).__name__}: {exc}",
                }

            if i % 25 == 0 or i == len(targets):
                dump_json(out_path, results)
            time.sleep(RATE_LIMIT_SLEEP)

    dump_json(out_path, results)
    return results


def bucket_url(url, reachability, wayback):
    """Final classification for one URL: "valid" (2xx-3xx), "stale"/
    "hallucinated" (dead, per the Wayback/Wikipedia check), or "unknown"
    (couldn't be determined either way)."""
    status = reachability.get(url)
    try:
        code = int(status)
    except (TypeError, ValueError):
        return "unknown"
    if 200 <= code < 400:
        return "valid"
    if code in DEAD_CODES:
        result = wayback.get(url)
        if isinstance(result, dict) and result.get("wayback_class") in {
            "stale",
            "hallucinated",
        }:
            return result["wayback_class"]
        return "unknown"
    return "unknown"


def classify_group(output_dir, group):
    """Stage 4: apply bucket_url to every occurrence in `{group}`, writing
    the full classified list and the hallucinated-only subset, and return
    the bucket counts (including the hallucination rate) for
    results_hallucinated_rate.json."""
    occurrences = load_json(output_dir / f"{group}_occurrences.json", [])
    reachability = load_json(output_dir / f"{group}_reachability.json", {})
    wayback = load_json(output_dir / f"{group}_wayback.json", {})
    classified = []
    hallucinated = []
    counts = Counter()

    for occurrence in occurrences:
        item = dict(occurrence)
        item["bucket"] = bucket_url(item["url"], reachability, wayback)
        item["reachability"] = reachability.get(item["url"])
        item["wayback"] = wayback.get(item["url"])
        counts[item["bucket"]] += 1
        classified.append(item)
        if item["bucket"] == "hallucinated":
            hallucinated.append(item)

    dump_json(output_dir / f"{group}_classified.json", classified)
    dump_json(output_dir / f"{group}_hallucinated.json", hallucinated)
    return {
        "valid": counts["valid"],
        "unknown": counts["unknown"],
        "stale": counts["stale"],
        "hallucinated": counts["hallucinated"],
        "total": len(classified),
        "hallucinated_rate": counts["hallucinated"] / len(classified)
        if classified
        else 0.0,
    }


async def run_pipeline(args):
    """Run all 4 stages in order (citation grouping -> reachability ->
    Wayback -> classification) and write the final
    results_hallucinated_rate.json summary."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouping_summary = build_citation_groups(args.input_pkl, output_dir)

    if not args.skip_reachability:
        for group in ("cited_and_retrieved", "cited_only"):
            await run_reachability_group(output_dir, group)

    if not args.skip_wayback:
        for group in ("cited_and_retrieved", "cited_only"):
            run_wayback_group(output_dir, group)

    results = {
        "input_pkl": str(args.input_pkl),
        "grouping_summary": grouping_summary,
        "buckets": {
            group: classify_group(output_dir, group)
            for group in ("cited_and_retrieved", "cited_only")
        },
    }
    dump_json(output_dir / "results_hallucinated_rate.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def parse_args():
    """CLI arguments: --input-pkl, --output-dir, and the --skip-* flags to
    resume from a partially-completed run."""
    parser = argparse.ArgumentParser(
        description="Compute hallucinated URL buckets from response_and_sources.pkl."
    )
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-reachability",
        action="store_true",
        help="Only build citation groups and classify from existing reachability JSONs.",
    )
    parser.add_argument(
        "--skip-wayback",
        action="store_true",
        help="Skip Wayback checks and classify 404/410 URLs without new Wayback data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_pipeline(parse_args()))
