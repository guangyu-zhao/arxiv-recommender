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
    ):
        self.max_paper_num = max_paper_num
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

    def get_response(self, title, abstract):
        prompt = """
            你是一个有帮助的学术研究助手，可以帮助我构建每日论文推荐系统。
            以下是我最近研究领域的描述：
            {}
        """.format(self.description)
        prompt += """
            以下是我从昨天的 arXiv 爬取的论文，我为你提供了标题和摘要：
            标题: {}
            摘要: {}
        """.format(title, abstract)
        prompt += """
            1. 总结这篇论文的主要内容。
            2. 请评估这篇论文与我研究领域的相关性，并给出 0-10 的评分。其中 0 表示完全不相关，10 表示高度相关。
            
            请按以下 JSON 格式给出你的回答：
            {
                "summary": <你的总结>,
                "relevance": <你的评分>
            }
            使用中文回答。
            直接返回上述 JSON 格式，无需任何额外解释。
        """

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
        prompt = f"""你是一个专业的学术论文解读助手。以下是一篇论文的内容：

【标题】
{title}

【摘要】
{abstract}

【论文正文节选】
{fulltext}

请从以下四个维度对这篇论文进行简明深入的解读，每个维度用 2~3 句话概括，使用中文：

1. 核心问题：这篇论文瞄准了什么科学或工程挑战？
2. 方法创新：提出了哪些新方法或新模块，有何独特的设计思路？
3. 实验结果：主要量化指标和结论是什么，相比已有方法性能提升幅度如何？
4. 局限与展望：作者指出了哪些局限性，以及未来的研究方向是什么？

请直接输出解读内容，无需额外格式或前缀。"""
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

            fulltext = get_paper_fulltext(arxiv_id)
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
        )[: self.max_paper_num]

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
        prompt_context = """
            你是一个有帮助的学术研究助手，可以帮助我构建每日论文推荐系统。
            以下是我最近研究领域的描述：
            {}
        """.format(self.description)
        papers_context = """
            以下是我从昨天的 arXiv 爬取的论文，我为你提供了标题和摘要：
            {}
        """.format(overview)
        json_instruction = """
            请务必严格按照以下 JSON 结构返回内容，不要添加额外文本或代码块：
            {{
              "trend_summary": "<总体趋势，用中文,使用 html 的语法，不要使用 markdown 的语法>",
              "recommendations": [
                {{
                  "title": "<论文标题>",
                  "relevance_label": "<高度相关/相关/一般相关>",
                  "recommend_reason": "<为什么值得我读>",
                  "key_contribution": "<一句话概括论文关键贡献>"
                }}
              ],
              "additional_observation": "<补充观察，若无请写‘暂无’>"
            }}

            任务要求：
            1. 给出今天论文体现的整体研究趋势，解释其与我研究兴趣的联系。
            2. 精选最值得我精读的论文（建议返回 3-5 篇，可按实际情况增减），说明推荐理由并突出关键贡献。
            3. 如有需要持续关注或潜在风险的方向，请在补充观察中说明；若没有请写“暂无”。
        """
        html_instruction = """
            请直接输出一段 HTML 片段，严格遵循以下结构，不要包含 JSON、Markdown 或多余说明：
            <div class="summary-wrapper">
              <div class="summary-section">
                <h2>今日研究趋势</h2>
                <p>...</p>
              </div>
              <div class="summary-section">
                <h2>重点推荐</h2>
                <ol class="summary-list">
                  <li class="summary-item">
                    <div class="summary-item__header"><span class="summary-item__title">论文标题</span><span class="summary-pill">相关性</span></div>
                    <p><strong>推荐理由：</strong>...</p>
                    <p><strong>关键贡献：</strong>...</p>
                  </li>
                </ol>
              </div>
              <div class="summary-section">
                <h2>补充观察</h2>
                <p>暂无或其他补充。</p>
              </div>
            </div>

            HTML 要用中文撰写内容，重点推荐部分建议返回 3-5 篇论文，可按实际情况增减，缺少推荐时请写“暂无推荐。”。
        """
        prompt = prompt_context + papers_context + json_instruction
        html_prompt = prompt_context + papers_context + html_instruction

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
        receiver: str,
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

        # 处理多个接收者
        receivers = [addr.strip() for addr in receiver.split(",")]
        print(receivers)
        msg["To"] = ",".join([_format_addr(f"You <%s>" % addr) for addr in receivers])

        today = self.run_datetime.strftime("%Y/%m/%d")
        msg["Subject"] = Header(f"{title} {today}", "utf-8").encode()

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        except Exception as e:
            logger.warning(f"Failed to use TLS. {e}")
            logger.warning(f"Try to use SSL.")
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
