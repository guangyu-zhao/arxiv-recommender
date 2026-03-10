from arxiv_daily import ArxivDaily
from llm import GPT
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arxiv Daily")
    parser.add_argument("--categories", nargs="+", required=True, help="arXiv categories, e.g. cs.CV cs.AI")
    parser.add_argument("--max_paper_num", type=int, default=20, help="Max papers to recommend")
    parser.add_argument("--max_entries", type=int, default=100, help="Max entries to fetch per category")

    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. gpt-4o")
    parser.add_argument("--base_url", type=str, required=True, help="API base URL, e.g. https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str, required=True, help="API key")

    parser.add_argument("--save", action="store_true", help="Save results to local files")
    parser.add_argument("--save_dir", type=str, default="./arxiv_history")
    parser.add_argument("--description", type=str, default="description.txt", help="Path to research interests file")

    parser.add_argument("--smtp_server", type=str, required=True, help="SMTP server host")
    parser.add_argument("--smtp_port", type=int, required=True, help="SMTP server port")
    parser.add_argument("--sender", type=str, required=True, help="Sender email address")
    parser.add_argument("--receiver", type=str, required=True, help="Receiver email(s), comma-separated")
    parser.add_argument("--sender_password", type=str, required=True, help="SMTP auth password")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--title", type=str, default="Daily arXiv", help="Email subject prefix")

    args = parser.parse_args()

    with open(args.description, "r") as f:
        args.description = f.read()

    # Test LLM availability
    try:
        model = GPT(args.model, args.base_url, args.api_key)
        model.inference("Hello, who are you?")
    except Exception as e:
        print(e)
        raise SystemExit("Failed to connect to LLM. Check your base_url, model, and api_key.")

    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = None

    arxiv_daily = ArxivDaily(
        args.categories,
        args.max_entries,
        args.max_paper_num,
        args.model,
        args.base_url,
        args.api_key,
        args.description,
        args.num_workers,
        args.temperature,
        args.save_dir,
    )

    arxiv_daily.send_email(
        args.sender,
        args.receiver,
        args.sender_password,
        args.smtp_server,
        args.smtp_port,
        args.title,
    )
