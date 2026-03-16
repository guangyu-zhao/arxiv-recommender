from llm import GPT
from util.request import get_yesterday_arxiv_papers
from util.construct_email import *
from tqdm import tqdm
import json
import os
from datetime import datetime, timezone
import time
import random
import smtplib
from email.header import Header
from email.utils import parseaddr, formataddr
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ArxivDaily:
    def __init__(
        self,
        categories: list[str],
        max_entries: int,
        max_paper_num: int,
        model: str,
        base_url: str,
        api_key: str,
        description: str,
        num_workers: int,
        temperature: float,
        save_dir: str | None,
        relevance_score_threshold: float = 7,
        fulltext_max_chars: int = 200000,
    ):
        self.max_paper_num = max_paper_num
        self.relevance_score_threshold = relevance_score_threshold
        self.fulltext_max_chars = fulltext_max_chars
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.temperature = temperature
        self.run_datetime = datetime.now(timezone.utc)
        self.run_date = self.run_datetime.strftime("%Y-%m-%d")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, save_dir, self.run_date, "json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.papers = {}
        for category in categories:
            self.papers[category] = get_yesterday_arxiv_papers(category, max_entries)
            print(
                "{} papers on arXiv for {} are fetched.".format(
                    len(self.papers[category]), category
                )
            )
            sleep_time = random.randint(5, 15)
            time.sleep(sleep_time)

        self.model = GPT(model, base_url, api_key)
        print(f"Model initialized successfully. Using {model} at {base_url}.")

        self.description = description
        self.lock = threading.Lock()

        # Load prompt templates
        prompt_dir = os.path.join(base_dir, "prompt")
        def _load_prompt(name):
            with open(os.path.join(prompt_dir, name), "r", encoding="utf-8") as f:
                return f.read()
        self._tpl_paper_scoring = _load_prompt("paper_scoring.txt")
        self._tpl_full_analysis = _load_prompt("full_analysis.txt")
        self._tpl_summarize_json = _load_prompt("summarize_json.txt")
        self._tpl_summarize_html = _load_prompt("summarize_html.txt")

    def get_response(self, title, abstract):
        prompt = self._tpl_paper_scoring.format(
            description=self.description, title=title, abstract=abstract
        )
        response = self.model.inference(prompt, temperature=self.temperature)
        return response

    def process_paper(self, paper, max_retries=5):
        retry_count = 0
        cache_path = os.path.join(self.cache_dir, f"{paper['arXiv_id']}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as cache_file:
                    cached_result = json.load(cache_file)
                print(f"缓存文件 {cache_path} 读取成功。")
                return cached_result
            except (json.JSONDecodeError, OSError) as e:
                print(f"缓存文件 {cache_path} 读取失败: {e}，将重新获取。")

        while retry_count < max_retries:
            try:
                title = paper["title"]
                abstract = paper["abstract"]
                response = self.get_response(title, abstract)
                response = response.strip("```").strip("json")
                response = json.loads(response)
                relevance_score = float(response["relevance"])
                summary = response["summary"]
                result = {
                    "title": title,
                    "arXiv_id": paper["arXiv_id"],
                    "abstract": abstract,
                    "summary": summary,
                    "relevance_score": relevance_score,
                    "pdf_url": paper["pdf_url"],
                }
                try:
                    with self.lock:
                        with open(cache_path, "w", encoding="utf-8") as cache_file:
                            json.dump(result, cache_file, ensure_ascii=False, indent=2)
                except OSError as write_error:
                    print(f"写入缓存 {cache_path} 时失败: {write_error}")
                return result
            except Exception as e:
                retry_count += 1
                print(f"处理论文 {paper['arXiv_id']} 时发生错误: {e}")
                print(f"正在进行第 {retry_count} 次重试...")
                if retry_count == max_retries:
                    print(f"已达到最大重试次数 {max_retries}，放弃处理论文{paper['arXiv_id']}")
                    # 处理失败，返回特殊结果
                    result = {
                        "title": paper["title"],
                        "arXiv_id": paper["arXiv_id"],
                        "abstract": paper["abstract"],
                        "summary": "该论文总结失败",
                        "relevance_score": 10,
                        "pdf_url": paper.get("pdf_url", ""),
                    }
                    try:
                        with self.lock:
                            with open(cache_path, "w", encoding="utf-8") as cache_file:
                                json.dump(result, cache_file, ensure_ascii=False, indent=2)
                    except OSError as write_error:
                        print(f"写入缓存 {cache_path} 时失败: {write_error}")
                    return result
                time.sleep(1)  # 重试前等待1秒

    def get_full_analysis(self, title: str, abstract: str, fulltext: str) -> str:
        prompt = self._tpl_full_analysis.format(
            title=title, abstract=abstract, fulltext=fulltext
        )
        return self.model.inference(prompt, temperature=self.temperature)

    def enrich_with_fulltext(self, recommendations: list) -> list:
        """为 top-N 推荐论文爬取全文并生成详细解读，结果写入独立缓存文件。"""
        from util.request import get_paper_fulltext

        print("Fetching full text and generating detailed analysis...")
        for paper in tqdm(recommendations, desc="Full-text analysis"):
            arxiv_id = paper["arXiv_id"]
            cache_path = os.path.join(self.cache_dir, f"{arxiv_id}_fulltext.json")

            # 优先读缓存
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    paper["full_analysis"] = cached.get("full_analysis", "")
                    print(f"全文分析缓存命中: {arxiv_id}")
                    continue
                except (json.JSONDecodeError, OSError) as e:
                    print(f"全文分析缓存读取失败: {e}，将重新获取。")

            fulltext = get_paper_fulltext(arxiv_id, max_chars=self.fulltext_max_chars)
            if fulltext:
                try:
                    analysis = self.get_full_analysis(paper["title"], paper["abstract"], fulltext)
                except Exception as e:
                    print(f"全文解读生成失败 ({arxiv_id}): {e}")
                    analysis = "（全文解读生成失败）"
            else:
                analysis = "（未能获取论文 HTML 全文，跳过详细解读）"

            paper["full_analysis"] = analysis

            try:
                with self.lock:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"full_analysis": analysis}, f, ensure_ascii=False, indent=2)
            except OSError as e:
                print(f"写入全文分析缓存失败 ({arxiv_id}): {e}")

        return recommendations

    def get_recommendation(self):
        recommendations = {}
        for category, papers in self.papers.items():
            for paper in papers:
                recommendations[paper["arXiv_id"]] = paper

        print(
            f"Got {len(recommendations)} non-overlapping papers from yesterday's arXiv."
        )

        recommendations_ = []
        print("Performing LLM inference...")

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            for arXiv_id, paper in recommendations.items():
                futures.append(executor.submit(self.process_paper, paper))
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing papers",
                unit="paper",
            ):
                result = future.result()
                if result:
                    recommendations_.append(result)

        recommendations_ = sorted(
            recommendations_, key=lambda x: x["relevance_score"], reverse=True
        )
        recommendations_ = [p for p in recommendations_ if p["relevance_score"] >= self.relevance_score_threshold][
            : self.max_paper_num
        ]

        # Save recommendation to markdown file
        current_time = self.run_datetime
        save_path = os.path.join(
            self.save_dir, self.run_date, f"{current_time.strftime('%Y-%m-%d')}.md"
        )
        with open(save_path, "w") as f:
            f.write("# Daily arXiv Papers\n")
            f.write(f"## Date: {current_time.strftime('%Y-%m-%d')}\n")
            f.write(f"## Description: {self.description}\n")
            f.write("## Papers:\n")
            for i, paper in enumerate(recommendations_):
                f.write(f"### {i + 1}. {paper['title']}\n")
                f.write(f"#### Abstract:\n")
                f.write(f"{paper['abstract']}\n")
                f.write(f"#### Summary:\n")
                f.write(f"{paper['summary']}\n")
                f.write(f"#### Relevance Score: {paper['relevance_score']}\n")
                f.write(f"#### PDF URL: {paper['pdf_url']}\n")
                f.write("\n")

        return recommendations_

    def summarize(self, recommendations):
        overview = ""
        for i in range(len(recommendations)):
            overview += f"{i + 1}. {recommendations[i]['title']} - {recommendations[i]['summary']} \n"
        prompt = self._tpl_summarize_json.format(
            description=self.description, overview=overview
        )
        html_prompt = self._tpl_summarize_html.format(
            description=self.description, overview=overview
        )

        def _clean_model_response(raw_text: str) -> str:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if "\n" in cleaned:
                    first_line, rest = cleaned.split("\n", 1)
                    if first_line.strip().lower() in ("json", "html"):
                        cleaned = rest
                    else:
                        cleaned = first_line + "\n" + rest
            return cleaned.strip()

        max_retries = 1
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = self.model.inference(
                    prompt, temperature=self.temperature
                )
                cleaned = _clean_model_response(raw_response)
                data = json.loads(cleaned)
                trend_summary = data.get("trend_summary", "暂无趋势信息")
                recommendations_data = data.get("recommendations", [])
                additional_observation = data.get("additional_observation", "暂无")

                if not isinstance(recommendations_data, list):
                    raise ValueError("recommendations 字段不是列表")

                cleaned_recommendations = []
                for item in recommendations_data:
                    title = item.get("title")
                    if not title:
                        raise ValueError("recommendations 中存在缺少标题的条目")
                    cleaned_recommendations.append(
                        {
                            "title": title,
                            "relevance_label": item.get(
                                "relevance_label", "相关性未知"
                            ),
                            "recommend_reason": item.get(
                                "recommend_reason", "未提供推荐理由"
                            ),
                            "key_contribution": item.get(
                                "key_contribution", "未提供关键贡献"
                            ),
                        }
                    )

                structured_summary = {
                    "trend_summary": trend_summary,
                    "recommendations": cleaned_recommendations,
                    "additional_observation": additional_observation,
                }

                return render_summary_sections(structured_summary)
            except Exception as error:
                print(f"总结生成第 {attempt} 次失败: {error}")
                if attempt == max_retries:
                    try:
                        for html_attempt in range(1, max_retries + 1):  
                            print(f"HTML 回退生成第 {html_attempt} 次...")
                            raw_html_response = self.model.inference(
                                html_prompt, temperature=self.temperature
                            )
                            cleaned_html = _clean_model_response(raw_html_response)
                            return cleaned_html
                    except Exception as html_error:
                        print(f"HTML 回退生成失败: {html_error}")
                        fallback_data = {
                            "trend_summary": "总结生成失败，请稍后重试。",
                            "recommendations": [],
                            "additional_observation": "暂无。",
                        }
                        return render_summary_sections(fallback_data)

    def render_email(self, recommendations):
        save_file_path = os.path.join(self.save_dir, self.run_date, "arxiv_daily_email.html")
        if os.path.exists(save_file_path):
            with open(save_file_path, "r", encoding="utf-8") as f:
                print(f"邮件已渲染，从缓存文件 {save_file_path} 读取邮件。")
                return f.read()
        parts = []
        if len(recommendations) == 0:
            return framework.replace("__CONTENT__", get_empty_html())
        for i, p in enumerate(tqdm(recommendations, desc="Rendering Emails")):
            rate = get_stars(p["relevance_score"])
            parts.append(
                get_block_html(
                    str(i + 1) + ". " + p["title"],
                    rate,
                    p["arXiv_id"],
                    p["summary"],
                    p["pdf_url"],
                    p.get("full_analysis", ""),
                )
            )
        summary = self.summarize(recommendations)
        # Add the summary to the start of the email
        content = summary
        content += "<br>" + "</br><br>".join(parts) + "</br>"
        email_html = framework.replace("__CONTENT__", content)
        # 保存渲染后的邮件到 save_dir
        if self.save_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(base_dir, self.save_dir, self.run_date, "arxiv_daily_email.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(email_html)
        return email_html

    def send_email(
        self,
        sender: str,
        receivers: list[str],
        password: str,
        smtp_server: str,
        smtp_port: int,
        title: str,
    ):
        recommendations = self.get_recommendation()
        recommendations = self.enrich_with_fulltext(recommendations)
        html = self.render_email(recommendations)

        def _format_addr(s):
            name, addr = parseaddr(s)
            return formataddr((Header(name, "utf-8").encode(), addr))

        msg = MIMEText(html, "html", "utf-8")
        msg["From"] = _format_addr(f"{title} <%s>" % sender)

        msg["To"] = ",".join([_format_addr(f"You <%s>" % addr) for addr in receivers])

        today = self.run_datetime.strftime("%Y/%m/%d")
        msg["Subject"] = Header(f"{title} {today}", "utf-8").encode()

        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()


if __name__ == "__main__":
    categories = ["cs.CV"]
    max_entries = 100
    max_paper_num = 50
    model = "deepseek-ai/DeepSeek-V3"
    base_url = "https://api.siliconflow.cn/v1"
    api_key = "YOUR_API_KEY"
    description = """
        I am working on the research area of computer vision and natural language processing. 
        Specifically, I am interested in the following fieds:
        1. Object detection
        2. AIGC (AI Generated Content)
        3. Multimodal Large Language Models

        I'm not interested in the following fields:
        1. 3D Vision
        2. Robotics
        3. Low-level Vision
    """

    arxiv_daily = ArxivDaily(
        categories, max_entries, max_paper_num,
        model, base_url, api_key, description,
        num_workers=4, temperature=0.7, save_dir="./arxiv_history",
    )
    recommendations = arxiv_daily.get_recommendation()
    print(recommendations)
