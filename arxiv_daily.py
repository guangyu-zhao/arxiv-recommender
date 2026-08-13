from llm import GPT
from util.request import get_yesterday_arxiv_papers
from util.construct_email import *
from tqdm import tqdm
import json
import os
from datetime import datetime, timezone
import fcntl
import tempfile
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
        profile: str,
        relevance_score_threshold: float = 7,
        fulltext_max_chars: int = 200000,
        llm_requests_per_minute: float | None = 30,
        llm_max_attempts: int = 6,
        llm_retry_base_seconds: float = 2,
        arxiv_timeout_seconds: float = 30,
        arxiv_max_attempts: int = 5,
    ):
        self.max_paper_num = max_paper_num
        self.relevance_score_threshold = relevance_score_threshold
        self.fulltext_max_chars = fulltext_max_chars
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.temperature = temperature
        self.arxiv_timeout_seconds = arxiv_timeout_seconds
        self.arxiv_max_attempts = arxiv_max_attempts
        self.run_datetime = datetime.now(timezone.utc)
        self.run_date = self.run_datetime.strftime("%Y-%m-%d")
        profile_name = profile.strip()
        if not profile_name:
            profile_name = "default"
        self.profile = "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "_"
            for ch in profile_name
        )
        if not self.profile:
            self.profile = "default"
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_root = None
        self.profile_root = None
        if self.save_dir:
            expanded_save_dir = os.path.expanduser(self.save_dir)
            if os.path.isabs(expanded_save_dir):
                self.save_root = expanded_save_dir
            else:
                self.save_root = os.path.join(self.base_dir, expanded_save_dir)
            self.profile_root = os.path.join(self.save_root, self.profile)
            os.makedirs(self.profile_root, exist_ok=True)
        self.cache_dir = None
        if self.profile_root:
            self.cache_dir = os.path.join(self.profile_root, self.run_date, "json")
            os.makedirs(self.cache_dir, exist_ok=True)
        self.papers = {}

        # Load seen arXiv IDs to avoid duplicate processing
        self.seen_ids = set()
        self.successfully_processed_ids = set()
        self.pending_papers = {}
        if self.profile_root:
            self.seen_ids_path = os.path.join(self.profile_root, "seen_arxiv_ids.json")
            self.pending_papers_path = os.path.join(
                self.profile_root, "pending_arxiv_papers.json"
            )
            if os.path.exists(self.seen_ids_path):
                try:
                    with open(self.seen_ids_path, "r", encoding="utf-8") as f:
                        self.seen_ids = set(json.load(f))
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Failed to load seen_arxiv_ids.json: {e}")
            if os.path.exists(self.pending_papers_path):
                try:
                    with open(self.pending_papers_path, "r", encoding="utf-8") as f:
                        pending_items = json.load(f)
                    self.pending_papers = {
                        paper["arXiv_id"]: paper
                        for paper in pending_items
                        if paper["arXiv_id"] not in self.seen_ids
                    }
                except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
                    raise RuntimeError(
                        f"Failed to load pending_arxiv_papers.json: {e}"
                    ) from e

        # Keep current-run IDs separate from durable seen IDs. Durable state is
        # committed only after the email has been delivered successfully.
        current_run_ids = set(self.pending_papers)
        if self.pending_papers:
            self.papers["pending"] = list(self.pending_papers.values())
            print(f"Loaded {len(self.pending_papers)} pending papers for retry.")
        for category in categories:
            fetched_papers = get_yesterday_arxiv_papers(
                category,
                max_entries,
                timeout=self.arxiv_timeout_seconds,
                max_attempts=self.arxiv_max_attempts,
            )
            new_papers = []
            for paper in fetched_papers:
                arxiv_id = paper["arXiv_id"]
                if arxiv_id not in self.seen_ids and arxiv_id not in current_run_ids:
                    new_papers.append(paper)
                    current_run_ids.add(arxiv_id)
                    self.pending_papers[arxiv_id] = paper

            self.papers[category] = new_papers
            if self.profile_root and new_papers:
                # Checkpoint after each category so a later category/network
                # failure cannot discard candidates already fetched.
                self._write_json_atomically(
                    self.pending_papers_path,
                    list(self.pending_papers.values()),
                )
            print(
                "{} new papers on arXiv for {} are fetched (out of {} total).".format(
                    len(self.papers[category]), category, len(fetched_papers)
                )
            )
            sleep_time = random.randint(5, 15)
            time.sleep(sleep_time)

        # Persist the retry queue before any LLM work starts. A crash from this
        # point onward can therefore be retried even after arXiv /new changes.
        if self.profile_root:
            self._write_json_atomically(
                self.pending_papers_path,
                list(self.pending_papers.values()),
            )

        self.model = GPT(
            model,
            base_url,
            api_key,
            requests_per_minute=llm_requests_per_minute,
            max_attempts=llm_max_attempts,
            retry_base_seconds=llm_retry_base_seconds,
        )
        print(f"Model initialized successfully. Using {model} at {base_url}.")

        self.description = description
        self.lock = threading.Lock()

        # Load prompt templates
        prompt_dir = os.path.join(self.base_dir, "prompt")
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

    @staticmethod
    def _write_json_atomically(path: str, value) -> None:
        target_dir = os.path.dirname(path)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=target_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                json.dump(value, temp_file, ensure_ascii=False, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _parse_scoring_response(raw_response: str) -> dict:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if "\n" in cleaned:
                first_line, remainder = cleaned.split("\n", 1)
                if first_line.strip().lower() == "json":
                    cleaned = remainder.strip()
        response = json.loads(cleaned)
        relevance_score = float(response["relevance"])
        if not 0 <= relevance_score <= 10:
            raise ValueError(f"relevance score out of range: {relevance_score}")
        summary = response["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary is empty")
        return {"summary": summary, "relevance_score": relevance_score}

    def process_paper(self, paper, max_parse_attempts=3):
        cache_path = (
            os.path.join(self.cache_dir, f"{paper['arXiv_id']}.json")
            if self.cache_dir
            else None
        )

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as cache_file:
                    cached_result = json.load(cache_file)
                if cached_result.get("summary") == "该论文总结失败":
                    raise ValueError("legacy failed-result cache")
                relevance_score = float(cached_result["relevance_score"])
                if not 0 <= relevance_score <= 10:
                    raise ValueError("cached relevance score is out of range")
                print(f"缓存文件 {cache_path} 读取成功。")
                return cached_result
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
                print(f"缓存文件 {cache_path} 读取失败: {e}，将重新获取。")

        for parse_attempt in range(1, max_parse_attempts + 1):
            try:
                title = paper["title"]
                abstract = paper["abstract"]
                response = self.get_response(title, abstract)
            except Exception as e:
                print(f"论文 {paper['arXiv_id']} 的 LLM 请求最终失败: {e}")
                return None

            try:
                parsed = self._parse_scoring_response(response)
                result = {
                    "title": title,
                    "arXiv_id": paper["arXiv_id"],
                    "abstract": abstract,
                    "summary": parsed["summary"],
                    "relevance_score": parsed["relevance_score"],
                    "pdf_url": paper["pdf_url"],
                }
                if cache_path:
                    try:
                        with self.lock:
                            with open(cache_path, "w", encoding="utf-8") as cache_file:
                                json.dump(result, cache_file, ensure_ascii=False, indent=2)
                    except OSError as write_error:
                        print(f"写入缓存 {cache_path} 时失败: {write_error}")
                return result
            except Exception as e:
                print(
                    f"论文 {paper['arXiv_id']} 的评分响应解析失败 "
                    f"({parse_attempt}/{max_parse_attempts}): {e}"
                )
                if parse_attempt < max_parse_attempts:
                    time.sleep(min(4, 2 ** (parse_attempt - 1)))

        print(f"放弃处理格式持续无效的论文 {paper['arXiv_id']}，稍后运行会重试。")
        return None

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
            cache_path = (
                os.path.join(self.cache_dir, f"{arxiv_id}_fulltext.json")
                if self.cache_dir
                else None
            )

            # 优先读缓存
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    paper["full_analysis"] = cached.get("full_analysis", "")
                    print(f"全文分析缓存命中: {arxiv_id}")
                    continue
                except (json.JSONDecodeError, OSError) as e:
                    print(f"全文分析缓存读取失败: {e}，将重新获取。")

            fulltext = get_paper_fulltext(
                arxiv_id,
                max_chars=self.fulltext_max_chars,
                timeout=self.arxiv_timeout_seconds,
                max_attempts=self.arxiv_max_attempts,
            )
            if fulltext:
                try:
                    analysis = self.get_full_analysis(paper["title"], paper["abstract"], fulltext)
                except Exception as e:
                    print(f"全文解读生成失败 ({arxiv_id}): {e}")
                    analysis = "（全文解读生成失败）"
            else:
                analysis = "（未能获取论文 HTML 全文，跳过详细解读）"

            paper["full_analysis"] = analysis

            if cache_path:
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
                    self.successfully_processed_ids.add(result["arXiv_id"])

        recommendations_ = sorted(
            recommendations_, key=lambda x: x["relevance_score"], reverse=True
        )
        recommendations_ = [p for p in recommendations_ if p["relevance_score"] >= self.relevance_score_threshold][
            : self.max_paper_num
        ]

        # Save recommendation to markdown file
        if self.profile_root:
            current_time = self.run_datetime
            save_path = os.path.join(
                self.profile_root, self.run_date, f"{current_time.strftime('%Y-%m-%d')}.md"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
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

    def commit_successfully_processed_ids(self) -> None:
        """Atomically mark papers seen only after the email was delivered."""
        if not self.profile_root or not self.successfully_processed_ids:
            return

        lock_path = os.path.join(self.profile_root, ".arxiv_state.lock")
        with open(lock_path, "a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                latest_seen_ids = set()
                if os.path.exists(self.seen_ids_path):
                    with open(self.seen_ids_path, "r", encoding="utf-8") as seen_file:
                        latest_seen_ids = set(json.load(seen_file))
                updated_seen_ids = latest_seen_ids | self.successfully_processed_ids
                self._write_json_atomically(
                    self.seen_ids_path,
                    sorted(updated_seen_ids),
                )

                latest_pending = dict(self.pending_papers)
                if os.path.exists(self.pending_papers_path):
                    with open(
                        self.pending_papers_path, "r", encoding="utf-8"
                    ) as pending_file:
                        latest_pending.update(
                            {
                                paper["arXiv_id"]: paper
                                for paper in json.load(pending_file)
                            }
                        )
                remaining_pending = {
                    arxiv_id: paper
                    for arxiv_id, paper in latest_pending.items()
                    if arxiv_id not in updated_seen_ids
                }
                self._write_json_atomically(
                    self.pending_papers_path,
                    list(remaining_pending.values()),
                )
                self.seen_ids = updated_seen_ids
                self.pending_papers = remaining_pending
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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
        save_file_path = None
        manifest_path = None
        if self.profile_root:
            save_file_path = os.path.join(self.profile_root, self.run_date, "arxiv_daily_email.html")
            manifest_path = os.path.join(
                self.profile_root,
                self.run_date,
                "arxiv_daily_email_manifest.json",
            )
        recommendation_ids = [paper["arXiv_id"] for paper in recommendations]
        if (
            save_file_path
            and manifest_path
            and os.path.exists(save_file_path)
            and os.path.exists(manifest_path)
        ):
            try:
                with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                    cached_ids = json.load(manifest_file)
                if cached_ids == recommendation_ids:
                    with open(save_file_path, "r", encoding="utf-8") as email_file:
                        print(f"邮件已渲染，从缓存文件 {save_file_path} 读取邮件。")
                        return email_file.read()
            except (json.JSONDecodeError, OSError, TypeError):
                pass

        parts = []
        if len(recommendations) == 0:
            email_html = framework.replace("__CONTENT__", get_empty_html())
        else:
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
        if save_file_path and manifest_path:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            with open(save_file_path, "w", encoding="utf-8") as f:
                f.write(email_html)
            self._write_json_atomically(manifest_path, recommendation_ids)
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

        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as server:
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())

        self.commit_successfully_processed_ids()


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
        num_workers=4, temperature=0.7, save_dir="./arxiv_history", profile="demo",
    )
    recommendations = arxiv_daily.get_recommendation()
    print(recommendations)
