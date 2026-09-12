import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deploy.run_ragas_answer_trial import _JudgeDiagnostics, _inject_judge_thinking


class RagasAnswerDiagnosticsTests(unittest.TestCase):
    def test_disabled_thinking_is_injected_into_instructor_defaults(self):
        judge_llm = SimpleNamespace(client=SimpleNamespace(kwargs={}))

        _inject_judge_thinking(judge_llm, "disabled")

        self.assertEqual(
            judge_llm.client.kwargs,
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )

    def test_hook_events_preserve_request_and_partial_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judge.jsonl"
            diagnostics = _JudgeDiagnostics(path, run_id="run-test", config={"judge_max_tokens": 8192})
            token = diagnostics.bind(case_id="C001", metric="faithfulness", outer_attempt=1)
            try:
                diagnostics.on_completion_kwargs(
                    model="deepseek-v4-flash",
                    max_tokens=8192,
                    extra_body={"thinking": {"type": "disabled"}},
                    messages=[{"role": "user", "content": "冻结证据"}],
                )
                error = RuntimeError("incomplete structured output")
                error.last_completion = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
                diagnostics.on_completion_last_attempt(error, attempt_number=1, is_last_attempt=True)
            finally:
                diagnostics.reset(token)

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            request = next(event for event in events if event["event"] == "completion_kwargs")
            failure = next(event for event in events if event["event"] == "completion_last_attempt")
            self.assertEqual(request["call_context"]["case_id"], "C001")
            self.assertEqual(request["kwargs"]["max_tokens"], 8192)
            self.assertEqual(request["kwargs"]["extra_body"]["thinking"]["type"], "disabled")
            self.assertEqual(failure["error"]["attributes"]["last_completion"]["choices"][0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
