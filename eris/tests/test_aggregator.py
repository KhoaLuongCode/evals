import unittest
from unittest.mock import MagicMock
from eris.aggregator import Aggregator
from eris.debate_flow import DebateTopic, DebateRound
from eris.judge_module import DebateResult
from eris.llm_interface import LLMInterface

class MockLLM(LLMInterface):
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        return "Mock LLM Response"

class MockJudge(LLMInterface):
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        # Always returns the same winner
        return "Reasoning...\nWinner: ProLLM"

class TestAggregator(unittest.TestCase):
    def test_aggregator_run(self):
        # Setup
        topics = [DebateTopic("Mock topic #1"), DebateTopic("Mock topic #2")]
        pro_llm = MockLLM("ProLLM", "", "")
        con_llm = MockLLM("ConLLM", "", "")
        judge = [MockJudge("JudgeLLM", "", "")]  # ensemble with one judge
        aggregator = Aggregator(topics, judge)

        # Execute
        aggregator.run_debates(llm_pairs=[[pro_llm, con_llm]], num_debates=2)
        self.assertEqual(len(aggregator.results), 2, "Should have 2 debate results.")

        # Check win rates
        rates = aggregator.compute_win_rates()
        # ProLLM should always be the winner
        self.assertGreaterEqual(rates.get("ProLLM", 0), 1.0)

if __name__ == "__main__":
    unittest.main()
