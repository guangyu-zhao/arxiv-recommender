"""
Use requests and BeautifulSoup to get yesterday's arXiv papers.
"""

import requests
from bs4 import BeautifulSoup
import random
import time
from urllib.parse import urljoin


ARXIV_BASE_URL = "https://arxiv.org"
REQUEST_HEADERS = {
    "User-Agent": (
        "arxiv-recommender/0.1 "
        "(+https://github.com/guangyu-zhao/arxiv-recommender)"
    )
}
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


def _retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _get_with_retries(
    url: str,
    timeout: float = 30,
    max_attempts: int = 5,
) -> requests.Response:
    """GET an arXiv page with bounded exponential backoff."""
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        response = None
        try:
            response = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=(10, timeout),
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            status_code = response.status_code if response is not None else None
            retryable = status_code is None or status_code in RETRYABLE_STATUS_CODES
            if attempt >= attempts or not retryable:
                raise

            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                delay = retry_after + random.uniform(0, 1)
            else:
                delay = min(60.0, 2 ** (attempt - 1))
                delay *= random.uniform(0.8, 1.2)
            print(
                f"arXiv request failed ({attempt}/{attempts}) for {url}; "
                f"retrying in {delay:.1f}s: {error}"
            )
            time.sleep(delay)


def get_yesterday_arxiv_papers(
    category: str,
    max_results: int,
    timeout: float = 30,
    max_attempts: int = 5,
):
    url = f"{ARXIV_BASE_URL}/list/{category}/new?skip=0&show={max_results}"

    response = _get_with_retries(url, timeout=timeout, max_attempts=max_attempts)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find("dl", id="articles")
    if articles is None:
        page_text = soup.get_text(" ", strip=True).lower()
        if "no new submissions" in page_text or "no submissions" in page_text:
            return []
        raise ValueError(f"Unexpected arXiv listing page for category {category}")

    entry_tags = articles.find_all("dt")
    if not entry_tags:
        return []

    papers = []
    for entry_tag in entry_tags:
        details_tag = entry_tag.find_next_sibling("dd")
        abstract_link = entry_tag.find("a", title="Abstract")
        pdf_link = entry_tag.find("a", title="Download PDF")
        title_tag = (
            details_tag.find("div", class_="list-title") if details_tag else None
        )
        abstract_tag = details_tag.find("p", class_="mathjax") if details_tag else None

        if not all((details_tag, abstract_link, pdf_link, title_tag, abstract_tag)):
            print(f"Skipping a malformed arXiv entry in category {category}")
            continue

        abs_url = urljoin(ARXIV_BASE_URL, abstract_link.get("href", ""))
        pdf_url = urljoin(ARXIV_BASE_URL, pdf_link.get("href", ""))
        arxiv_id = pdf_url.rstrip("/").split("/")[-1].removesuffix(".pdf")
        comments_tag = details_tag.find("div", class_="list-comments")
        papers.append(
            {
                "title": title_tag.get_text(" ", strip=True)
                .replace("Title:", "", 1)
                .strip(),
                "arXiv_id": arxiv_id,
                "abstract": abstract_tag.get_text(" ", strip=True),
                "comments": (
                    comments_tag.get_text(" ", strip=True)
                    if comments_tag
                    else "No comments available"
                ),
                "pdf_url": pdf_url,
                "abstract_url": abs_url,
            }
        )

    if not papers:
        raise ValueError(f"All arXiv entries were malformed for category {category}")

    return papers


def get_paper_fulltext(
    arxiv_id: str,
    max_chars: int,
    timeout: float = 30,
    max_attempts: int = 5,
) -> str:
    """从 arXiv HTML 页面爬取论文全文，返回纯文本；若不可用则返回空字符串。"""
    url = f"{ARXIV_BASE_URL}/html/{arxiv_id}"
    try:
        response = _get_with_retries(url, timeout=timeout, max_attempts=max_attempts)
        soup = BeautifulSoup(response.text, "html.parser")

        # 移除无关标签
        for tag in soup(["script", "style", "nav", "header", "footer", "figure"]):
            tag.decompose()

        # 移除参考文献节（通常是标题含 "reference" 或 "bibliography" 的 section）
        for section in soup.find_all("section"):
            heading = section.find(["h1", "h2", "h3", "h4"])
            if heading and heading.get_text().strip().lower() in (
                "references", "bibliography", "acknowledgements", "acknowledgments"
            ):
                section.decompose()

        # 优先取 <article>，其次 <main>，最后 <body>
        article = soup.find("article") or soup.find("main") or soup.body
        if article is None:
            return ""

        text = article.get_text(separator="\n", strip=True)
        # 压缩连续空行
        lines = [line for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        return text[:max_chars]
    except (requests.RequestException, ValueError) as error:
        print(f"Failed to fetch arXiv HTML for {arxiv_id}: {error}")
        return ""


if __name__ == "__main__":
    papers = get_yesterday_arxiv_papers("cs.CV", 100)
    print(len(papers))
