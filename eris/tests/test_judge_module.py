import unittest
from unittest.mock import MagicMock
from eris.judge_module import judge_debate, DebateResult
from eris.debate_flow import DebateTopic, DebateRound
from eris.llm_interface import LLMInterface

class MockJudge(LLMInterface):
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        # Pretend we parse the transcript and return a forced winner
        return "Reasoning: This is a mock.\nWinner: MockWinner"

class TestJudgeModule(unittest.TestCase):
    def test_judge_debate_single_judge(self):
        judge = MockJudge("JudgeLLM", "", "")
        topic = DebateTopic("Mock Topic")
        transcript = [
            DebateRound("ProLLM", "Pro statement"),
            DebateRound("ConLLM", "Con statement")
        ]

        result = judge_debate(judge, topic, transcript)
        self.assertIsInstance(result, DebateResult)
        self.assertEqual(result.winner, "MockWinner", "Winner should be the forced 'MockWinner'")
        self.assertIn("Reasoning", result.judge_reasoning)

if __name__ == "__main__":
    unittest.main()
