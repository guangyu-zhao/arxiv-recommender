import importlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from arxiv_daily import ArxivDaily
from util.request import _get_with_retries, get_yesterday_arxiv_papers


PAPER = {
    "title": "A useful paper",
    "arXiv_id": "2608.00001",
    "abstract": "An abstract",
    "comments": "No comments",
    "pdf_url": "https://arxiv.org/pdf/2608.00001",
    "abstract_url": "https://arxiv.org/abs/2608.00001",
}


class ArxivDailyReliabilityTests(unittest.TestCase):
    def make_daily(self, save_dir, fetched_papers=None):
        model = MagicMock()
        with (
            patch(
                "arxiv_daily.get_yesterday_arxiv_papers",
                return_value=fetched_papers if fetched_papers is not None else [PAPER],
            ),
            patch("arxiv_daily.GPT", return_value=model),
            patch("arxiv_daily.time.sleep"),
        ):
            daily = ArxivDaily(
                categories=["cs.AI"],
                max_entries=100,
                max_paper_num=20,
                model="test-model",
                base_url="https://example.test/v1",
                api_key="test-key",
                description="robot learning",
                num_workers=1,
                temperature=0,
                save_dir=save_dir,
                profile="test",
                llm_requests_per_minute=0,
            )
        return daily, model

    def test_candidates_are_pending_until_success_is_committed(self):
        with tempfile.TemporaryDirectory() as save_dir:
            daily, model = self.make_daily(save_dir)
            model.inference.return_value = json.dumps(
                {"summary": "Relevant work", "relevance": 8}
            )

            pending_path = os.path.join(
                save_dir, "test", "pending_arxiv_papers.json"
            )
            seen_path = os.path.join(save_dir, "test", "seen_arxiv_ids.json")
            with open(pending_path, encoding="utf-8") as pending_file:
                self.assertEqual(json.load(pending_file), [PAPER])
            self.assertFalse(os.path.exists(seen_path))

            recommendations = daily.get_recommendation()
            self.assertEqual([paper["arXiv_id"] for paper in recommendations], [PAPER["arXiv_id"]])
            self.assertFalse(os.path.exists(seen_path))

            daily.commit_successfully_processed_ids()
            with open(seen_path, encoding="utf-8") as seen_file:
                self.assertEqual(json.load(seen_file), [PAPER["arXiv_id"]])
            with open(pending_path, encoding="utf-8") as pending_file:
                self.assertEqual(json.load(pending_file), [])

    def test_pending_paper_is_retried_after_listing_changes(self):
        with tempfile.TemporaryDirectory() as save_dir:
            profile_dir = os.path.join(save_dir, "test")
            os.makedirs(profile_dir)
            with open(
                os.path.join(profile_dir, "pending_arxiv_papers.json"),
                "w",
                encoding="utf-8",
            ) as pending_file:
                json.dump([PAPER], pending_file)

            daily, _ = self.make_daily(save_dir, fetched_papers=[])
            pending_ids = {
                paper["arXiv_id"]
                for papers in daily.papers.values()
                for paper in papers
            }
            self.assertEqual(pending_ids, {PAPER["arXiv_id"]})

    def test_completed_category_is_checkpointed_before_later_fetch_failure(self):
        with tempfile.TemporaryDirectory() as save_dir:
            with (
                patch(
                    "arxiv_daily.get_yesterday_arxiv_papers",
                    side_effect=[[PAPER], requests.ConnectionError("offline")],
                ),
                patch("arxiv_daily.GPT"),
                patch("arxiv_daily.time.sleep"),
            ):
                with self.assertRaises(requests.ConnectionError):
                    ArxivDaily(
                        categories=["cs.AI", "cs.LG"],
                        max_entries=100,
                        max_paper_num=20,
                        model="test-model",
                        base_url="https://example.test/v1",
                        api_key="test-key",
                        description="robot learning",
                        num_workers=1,
                        temperature=0,
                        save_dir=save_dir,
                        profile="test",
                    )

            pending_path = os.path.join(
                save_dir, "test", "pending_arxiv_papers.json"
            )
            with open(pending_path, encoding="utf-8") as pending_file:
                self.assertEqual(json.load(pending_file), [PAPER])

    def test_failed_scoring_is_not_cached_or_assigned_a_score(self):
        with tempfile.TemporaryDirectory() as save_dir:
            daily, model = self.make_daily(save_dir)
            model.inference.side_effect = RuntimeError("provider unavailable")

            self.assertIsNone(daily.process_paper(PAPER))
            cache_path = os.path.join(
                daily.cache_dir, f"{PAPER['arXiv_id']}.json"
            )
            self.assertFalse(os.path.exists(cache_path))

    def test_invalid_scoring_response_is_never_promoted_to_ten(self):
        with tempfile.TemporaryDirectory() as save_dir:
            daily, model = self.make_daily(save_dir)
            model.inference.return_value = "not valid json"

            with patch("arxiv_daily.time.sleep"):
                self.assertIsNone(daily.process_paper(PAPER))

            self.assertEqual(model.inference.call_count, 3)
            self.assertEqual(daily.successfully_processed_ids, set())

    def test_smtp_failure_does_not_commit_seen_ids(self):
        with tempfile.TemporaryDirectory() as save_dir:
            daily, _ = self.make_daily(save_dir)
            daily.successfully_processed_ids.add(PAPER["arXiv_id"])
            daily.get_recommendation = MagicMock(return_value=[])
            daily.enrich_with_fulltext = MagicMock(return_value=[])
            daily.render_email = MagicMock(return_value="<html></html>")

            smtp_context = MagicMock()
            smtp_context.__enter__.return_value.login.side_effect = RuntimeError(
                "SMTP unavailable"
            )
            with patch("arxiv_daily.smtplib.SMTP_SSL", return_value=smtp_context):
                with self.assertRaisesRegex(RuntimeError, "SMTP unavailable"):
                    daily.send_email(
                        "sender@example.com",
                        ["receiver@example.com"],
                        "password",
                        "smtp.example.com",
                        465,
                        "Daily arXiv",
                    )

            self.assertFalse(os.path.exists(daily.seen_ids_path))

    def test_smtp_success_commits_seen_ids(self):
        with tempfile.TemporaryDirectory() as save_dir:
            daily, _ = self.make_daily(save_dir)
            daily.successfully_processed_ids.add(PAPER["arXiv_id"])
            daily.get_recommendation = MagicMock(return_value=[])
            daily.enrich_with_fulltext = MagicMock(return_value=[])
            daily.render_email = MagicMock(return_value="<html></html>")

            smtp_context = MagicMock()
            with patch("arxiv_daily.smtplib.SMTP_SSL", return_value=smtp_context):
                daily.send_email(
                    "sender@example.com",
                    ["receiver@example.com"],
                    "password",
                    "smtp.example.com",
                    465,
                    "Daily arXiv",
                )

            with open(daily.seen_ids_path, encoding="utf-8") as seen_file:
                self.assertEqual(json.load(seen_file), [PAPER["arXiv_id"]])

    def test_email_cache_is_reused_only_for_the_same_papers(self):
        with tempfile.TemporaryDirectory() as save_dir:
            daily, _ = self.make_daily(save_dir)
            recommendation = {
                **PAPER,
                "summary": "Relevant work",
                "relevance_score": 8,
                "full_analysis": "Analysis",
            }
            daily.summarize = MagicMock(return_value="<div>Summary</div>")

            first_html = daily.render_email([recommendation])
            second_html = daily.render_email([recommendation])
            empty_html = daily.render_email([])

            self.assertEqual(first_html, second_html)
            self.assertEqual(daily.summarize.call_count, 1)
            self.assertNotEqual(first_html, empty_html)


class LLMRetryTests(unittest.TestCase):
    def test_rate_limit_uses_bounded_backoff_then_succeeds(self):
        gpt_module = importlib.import_module("llm.GPT")

        error = RuntimeError("rate limited")
        error.status_code = 429
        success = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [error, success]

        with (
            patch.object(gpt_module, "OpenAI", return_value=client),
            patch.object(gpt_module.time, "sleep") as sleep,
            patch.object(gpt_module.random, "uniform", return_value=1),
        ):
            model = gpt_module.GPT(
                "model",
                "https://example.test/v1",
                "key",
                requests_per_minute=0,
                max_attempts=3,
                retry_base_seconds=2,
            )
            self.assertEqual(model.inference("hello"), "ok")

        self.assertEqual(client.chat.completions.create.call_count, 2)
        sleep.assert_called_once_with(2)


class ArxivRequestTests(unittest.TestCase):
    def test_network_error_is_retried_with_timeout_and_user_agent(self):
        response = MagicMock()
        response.status_code = 200
        response.text = "ok"

        with (
            patch(
                "util.request.requests.get",
                side_effect=[requests.ConnectionError("offline"), response],
            ) as request_get,
            patch("util.request.time.sleep") as sleep,
            patch("util.request.random.uniform", return_value=1),
        ):
            self.assertIs(_get_with_retries("https://arxiv.org/test"), response)

        self.assertEqual(request_get.call_count, 2)
        self.assertEqual(request_get.call_args.kwargs["timeout"], (10, 30))
        self.assertIn("User-Agent", request_get.call_args.kwargs["headers"])
        sleep.assert_called_once_with(1)

    def test_listing_parser_extracts_a_complete_paper(self):
        html = """
        <html><dl id="articles">
          <dt>
            <a title="Abstract" href="/abs/2608.00001">abs</a>
            <a title="Download PDF" href="/pdf/2608.00001">pdf</a>
          </dt>
          <dd>
            <div class="list-title">Title: A useful paper</div>
            <p class="mathjax">An abstract</p>
          </dd>
        </dl></html>
        """
        response = MagicMock(text=html)
        with patch("util.request._get_with_retries", return_value=response):
            papers = get_yesterday_arxiv_papers("cs.AI", 100)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["arXiv_id"], "2608.00001")
        self.assertEqual(papers[0]["title"], "A useful paper")


if __name__ == "__main__":
    unittest.main()
