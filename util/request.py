"""
Use requests and BeautifulSoup to get yesterday's arXiv papers.
"""

import requests
from bs4 import BeautifulSoup


def get_yesterday_arxiv_papers(category: str, max_results: int):
    url = f"https://arxiv.org/list/{category}/new?skip=0&show={max_results}"

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    try:
        entries = soup.find_all("dl", id="articles")[0].find_all(["dt", "dd"])
    except Exception as e:
        return []

    papers = []
    for i in range(0, len(entries), 2):
        title_tag = entries[i + 1].find("div", class_="list-title")
        title = (
            title_tag.text.strip().replace("Title:", "").strip()
            if title_tag
            else "No title available"
        )

        abs_url = "https://arxiv.org" + entries[i].find("a", title="Abstract")["href"]

        pdf_url = entries[i].find("a", title="Download PDF")["href"]
        pdf_url = "https://arxiv.org" + pdf_url

        abstract_tag = entries[i + 1].find("p", class_="mathjax")
        abstract = (
            abstract_tag.text.strip() if abstract_tag else "No abstract available"
        )

        comments_tag = entries[i + 1].find("div", class_="list-comments")
        comments = (
            comments_tag.text.strip() if comments_tag else "No comments available"
        )

        paper_info = {
            "title": title,
            "arXiv_id": pdf_url.split("/")[-1],
            "abstract": abstract,
            "comments": comments,
            "pdf_url": pdf_url,
            "abstract_url": abs_url,
        }

        papers.append(paper_info)

    return papers


def get_paper_fulltext(arxiv_id: str, max_chars: int) -> str:
    """从 arXiv HTML 页面爬取论文全文，返回纯文本；若不可用则返回空字符串。"""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return ""
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
    except Exception:
        return ""


if __name__ == "__main__":
    papers = get_yesterday_arxiv_papers("cs.CV", 100)
    print(len(papers))
