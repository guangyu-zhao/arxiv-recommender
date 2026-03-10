<h1 align="center">ArXiv-Recommender</h1>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README-zh.md">中文</a>
</p>

---

<p align="center">根据你的个性化描述，每天自动推荐你感兴趣的 arXiv 最新论文。</p>

> [!NOTE]
> 本项目借鉴了 [zotero-arxiv-daily](https://github.com/TideDra/zotero-arxiv-daily) 的思路和部分功能，感谢他们的出色工作！

## 核心特性

- 完全自定义的 LLM Prompt，引导论文推荐过程。
- 兼容**任何 OpenAI 格式的 API** —— OpenAI、DeepSeek、SiliconFlow、本地 Ollama 等。
- **全文深度解读**：对评分靠前的论文，自动爬取 arXiv HTML 全文，由 LLM 从四个维度进行深度解读（核心问题、方法创新、实验结果、局限与展望）。
- 邮件开头包含今日研究趋势总结和重点推荐论文。
- 将推荐历史保存为 Markdown 和 HTML 文件。
- 支持多线程并行加速推理过程。
- **结果缓存**：LLM 评分、全文解读、渲染后的邮件均按日期缓存，重复运行不会重复消耗 API 调用。
- 支持多收件人（逗号分隔）。

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/JoeLeelyf/customize-arxiv-daily.git
cd customize-arxiv-daily
pip install -r requirements.txt
```

或使用 [uv](https://github.com/astral-sh/uv)：

```bash
uv sync
```

### 2. 获取 SMTP 凭证

你需要一个支持 SMTP 的邮箱来发送推荐邮件。常见选择：

| 邮箱 | SMTP 服务器 | 端口 |
|------|-------------|------|
| QQ 邮箱 | `smtp.qq.com` | 465 |
| Gmail | `smtp.gmail.com` | 587 |
| 163 邮箱 | `smtp.163.com` | 465 |

> **注意**：`--sender_password` 填的**不是邮箱登录密码**，而是邮箱生成的 **SMTP 授权码**。以 QQ 邮箱为例：进入「设置 → 账户」，开启「POP3/SMTP 服务」，按提示验证后即可获得一个 16 位授权码。

### 3. 描述你的研究兴趣

编辑 `description.txt`，写下你感兴趣和不感兴趣的研究方向：

```txt
I am working on the research area of computer vision.
Specifically, I am interested in the following fields:
1. Object detection
2. AIGC (AI Generated Content)
3. Multimodal Large Language Models

I'm not interested in the following fields:
1. 3D Vision
2. Robotics
3. Low-level Vision
```

### 4. 运行

项目使用统一接口：`--base_url`、`--model` 和 `--api_key`，兼容任何 OpenAI 格式的 API。

**OpenAI：**

```bash
python main.py --categories cs.CV cs.AI \
    --model gpt-4o \
    --base_url https://api.openai.com/v1 --api_key 你的API密钥 \
    --smtp_server smtp.qq.com --smtp_port 465 \
    --sender 你的邮箱@qq.com --receiver 接收邮箱@gmail.com \
    --sender_password 你的SMTP授权码 \
    --num_workers 16 --temperature 0.7 --save
```

**SiliconFlow（DeepSeek 等国产模型）：**

```bash
python main.py --categories cs.CV cs.AI \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --base_url https://api.siliconflow.cn/v1 --api_key 你的API密钥 \
    --smtp_server smtp.qq.com --smtp_port 465 \
    --sender 你的邮箱@qq.com --receiver 接收邮箱@gmail.com \
    --sender_password 你的SMTP授权码 \
    --num_workers 16 --temperature 0.7 --save
```

**本地 Ollama（通过其 OpenAI 兼容端点）：**

```bash
python main.py --categories cs.CV cs.AI \
    --model deepseek-r1:7b \
    --base_url http://localhost:11434/v1 --api_key ollama \
    --smtp_server smtp.qq.com --smtp_port 465 \
    --sender 你的邮箱@qq.com --receiver 接收邮箱@gmail.com \
    --sender_password 你的SMTP授权码 \
    --num_workers 4 --temperature 0.7 --save
```

也可以直接使用项目提供的 shell 脚本：

```bash
bash main_gpt.sh
bash main_silicon_flow.sh
```

### 5.（可选）每天自动运行

在 Linux 上使用 `crontab` 设置定时任务：

```bash
crontab -e
```

添加以下行，每天早上 5:00 自动运行：

```txt
0 5 * * * /path/to/customize-arxiv-daily/main_gpt.sh
```

### 6.（可选）自定义 LLM Prompt

编辑 `arxiv_daily.py` 中的 `get_response()` 方法，调整 LLM 对每篇论文的评估方式；或编辑 `get_full_analysis()` 方法，修改全文深度解读的维度。

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--categories` | arXiv 分类，如 `cs.CV cs.AI` | 必填 |
| `--model` | 模型名称，如 `gpt-4o` | 必填 |
| `--base_url` | API 地址，如 `https://api.openai.com/v1` | 必填 |
| `--api_key` | API 密钥 | 必填 |
| `--smtp_server` | SMTP 服务器地址 | 必填 |
| `--smtp_port` | SMTP 服务器端口 | 必填 |
| `--sender` | 发件人邮箱 | 必填 |
| `--receiver` | 收件人邮箱，多个用逗号分隔 | 必填 |
| `--sender_password` | SMTP 授权码 | 必填 |
| `--max_paper_num` | 推荐论文数量上限 | 20 |
| `--max_entries` | 每个分类最多爬取论文数 | 100 |
| `--num_workers` | 并行线程数 | 4 |
| `--temperature` | LLM 采样温度 | 0.7 |
| `--title` | 邮件标题前缀 | `Daily arXiv` |
| `--save` | 是否保存结果到本地 | 关闭 |
| `--save_dir` | 保存目录 | `./arxiv_history` |
| `--description` | 研究兴趣描述文件路径 | `description.txt` |

## 工作原理

1. `util/request.py` 爬取指定分类的 arXiv "new" 页面，提取论文元数据（标题、摘要、PDF 链接等）。
2. `arxiv_daily.py` 通过 OpenAI 兼容 API 并行调用 LLM，对每篇论文进行总结并评估与研究兴趣的相关性（0–10 分）。结果按日期缓存到本地。
3. 按相关性排序后取前 N 篇论文。针对每篇论文，`util/request.py` 爬取 arXiv HTML 全文，LLM 从四个维度生成深度解读（核心问题、方法创新、实验结果、局限与展望），同样缓存到本地。
4. `util/construct_email.py` 渲染 HTML 邮件：顶部为今日研究趋势总结与重点推荐，之后是各论文卡片。邮件保存到本地并通过 SMTP 发送。

## 局限性

- LLM 的推荐过程具有不确定性，不同模型和不同运行之间的相关性评分可能存在较大差异。
- 全文深度解读依赖 arXiv 提供 HTML 版本的论文页面，部分较旧的论文可能没有 HTML 全文。

## 致谢

- [zotero-arxiv-daily](https://github.com/TideDra/zotero-arxiv-daily)
