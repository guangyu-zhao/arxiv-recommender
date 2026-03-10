cd /path/to/customized-arxiv-daily
python main.py --categories cs.CV cs.AI cs.CL cs.CR cs.LG \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --base_url https://api.siliconflow.cn/v1 --api_key YOUR_API_KEY \
    --smtp_server smtp.qq.com --smtp_port 465 \
    --sender YOUR_EMAIL@qq.com --receiver RECEIVER@gmail.com \
    --sender_password YOUR_SMTP_PASSWORD \
    --save
