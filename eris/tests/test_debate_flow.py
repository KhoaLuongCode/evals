import unittest
from unittest.mock import MagicMock
from eris.debate_flow import debate_transcript, DebateTopic, DebateRound
from eris.llm_interface import LLMInterface

class MockLLM(LLMInterface):
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        return f"Mock response to: {prompt[:30]}..."

class TestDebateFlow(unittest.TestCase):
    def test_debate_transcript(self):
        pro_llm = MockLLM("ProLLM", "", "")
        con_llm = MockLLM("ConLLM", "", "")
        topic = DebateTopic("Test Topic")

        rounds = debate_transcript(pro_llm, con_llm, topic)
        self.assertEqual(len(rounds), 8, "Should have 8 rounds in the standard debate format.")
        self.assertTrue(any("ProLLM" in r.speaker for r in rounds), "Transcript must contain ProLLM responses.")
        self.assertTrue(any("ConLLM" in r.speaker for r in rounds), "Transcript must contain ConLLM responses.")

if __name__ == "__main__":
    unittest.main()
