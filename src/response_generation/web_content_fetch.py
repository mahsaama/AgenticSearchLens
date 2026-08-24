"""Fetches and caches the raw text content of cited/retrieved source URLs.

Given a URL (news article, PDF, Wikipedia page, ...), tries progressively
heavier extraction strategies until one returns usable text: a plain
`requests` GET (fastest), the Wikipedia REST API for wikipedia.org links,
then a full Playwright browser render (cookie-banner/paywall dismissal,
live-DOM extraction) as the fallback for JS-heavy or blocked pages.
extract_urls_content() is the entry point: it reads every source URL cited
or retrieved across the sampled response_and_sources rows (see
_load_response_source_similarity_input()) and writes their fetched text to
a single on-disk cache (RESPONSE_URLS_CONTENT_PATH) so downstream analyses
(response_generation.py's response_source_similarity(), claim_extraction.py
and entailment_analysis.py's NLI/factuality pipelines) never re-fetch the
same URL twice.

Split out of response_generation.py, alongside claim_extraction.py and
entailment_analysis.py -- response_generation.py imports the entry points
it needs from all three rather than defining them itself.

Run directly (`python -m src.response_generation.web_content_fetch`) for a
small smoke test: fetches a couple of well-known, stable URLs and prints
how much text came back from each, without touching outputs/.
"""

import asyncio
import logging
import os
import re
from typing import List
from urllib.parse import unquote, urlparse

import ast
import fitz
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from readability import Document
from tqdm import tqdm

from src.utils.common_io import OUTPUT_PATH, load_json, to_json

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
    f"{OUTPUT_PATH}/chatgpt/metadata/response_and_sources_url_content.json"
)

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

def _load_response_source_similarity_input(platform="chatgpt"):
    """English rows from response_and_sources.pkl/csv, sampled up to 100
    per topic, preferring the paper's original cohort topics (Science,
    Health, Politics & History). If none of those topics are present in
    this platform's data -- e.g. a small non-paper sample -- falls back to
    sampling (same per-topic cap) across every topic actually present,
    rather than silently returning an empty frame."""
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

    df = df[df["language"] == "en"].copy()
    if "srcs_cited" in df.columns and "srcs_retrieved" in df.columns:
        has_image_url_mask = df.apply(_row_has_cited_or_retrieved_image_url, axis=1)
        df = df.loc[~has_image_url_mask].copy()

    # print(df["topic"].value_counts())

    def _sample_by_topic(candidate_df, topics):
        sampled_frames = []
        for topic in topics:
            topic_df = candidate_df[candidate_df["topic"] == topic].copy()
            if topic_df.empty:
                continue
            sample_n = min(100, len(topic_df))
            sampled_frames.append(topic_df.sample(n=sample_n, random_state=random_state))
        if not sampled_frames:
            return candidate_df.iloc[0:0].copy()
        return (
            pd.concat(sampled_frames, ignore_index=True)
            .sort_values(["topic", "conv_id", "turn_id"], kind="stable")
            .reset_index(drop=True)
        )

    sampled_df = _sample_by_topic(
        df[df["topic"].isin(selected_topics)].copy(), selected_topics
    )

    if len(sampled_df) == 0:
        # None of the paper's original cohort topics (Science/Health/
        # Politics & History) are present in this data -- e.g. a small
        # non-paper sample -- so fall back to sampling across every topic
        # actually present instead of silently returning nothing.
        all_topics = sorted(df["topic"].dropna().astype(str).unique().tolist())
        sampled_df = _sample_by_topic(df, all_topics)

    return sampled_df

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
    urls_content_path=None,
    force_refresh=False,
    platform="chatgpt",
):
    if urls_content_path is None:
        urls_content_path = (
            RESPONSE_URLS_CONTENT_PATH
            if platform == "chatgpt"
            else f"{OUTPUT_PATH}/{platform}/metadata/response_and_sources_url_content.json"
        )
    df = _load_response_source_similarity_input(platform=platform)

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

def _smoke_test():
    """Standalone sanity check: fetch a couple of stable, well-known URLs
    and report how much text came back from each. Doesn't touch outputs/ or
    require any pipeline data to exist first -- just exercises the fetch
    strategies (requests / Wikipedia API / Playwright fallback) against the
    live internet.
    """
    test_urls = [
        "https://en.wikipedia.org/wiki/Web_scraping",
        "https://www.python.org/about/",
    ]

    async def _run():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                for url in test_urls:
                    content = await fetch_url_content(url, browser=browser)
                    print(f"{url}: {len(content)} chars fetched")
            finally:
                await browser.close()

    asyncio.run(_run())


if __name__ == "__main__":
    _smoke_test()
