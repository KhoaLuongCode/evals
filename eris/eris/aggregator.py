import random
from typing import List, Dict
from dataclasses import dataclass, field

from debate_flow import DebateTopic, debate_transcript
from judge_module import judge_debate, DebateResult
from llm_interface import LLMInterface

@dataclass
class Aggregator:
    """
    Orchestrates multiple debate sessions among pairs or sets of LLMs,
    collects results, and computes stats.
    """
    topics: List[DebateTopic]
    judge_llms: List[LLMInterface]
    results: List[DebateResult] = field(default_factory=list)

    def run_debates(
        self,
        llm_pairs: List[List[LLMInterface]],
        num_debates: int = 5,
        max_tokens: int = 256,
        temperature: float = 0.7
    ):
        """
        For each pair of LLMs, randomly pick a topic and run debates 'num_debates' times.
        """
        for pair in llm_pairs:
            pro_llm, con_llm = pair
            for _ in range(num_debates):
                topic = random.choice(self.topics)
                # Run debate flow
                d_transcript = debate_transcript(pro_llm, con_llm, topic, max_tokens, temperature)
                # Now judge
                judged_result = judge_debate(
                    judge_llms=self.judge_llms,
                    topic=topic,
                    transcript=d_transcript,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                self.results.append(judged_result)

    def compute_win_rates(self) -> Dict[str, float]:
        """
        Computes how often each speaker is declared the winner
        across all stored results.
        """
        win_counts: Dict[str, int] = {}
        appear_counts: Dict[str, int] = {}

        for res in self.results:
            w = res.winner
            if w not in win_counts:
                win_counts[w] = 0
            win_counts[w] += 1

            speaker_set = {round_.speaker for round_ in res.transcript}
            for speaker in speaker_set:
                if speaker not in appear_counts:
                    appear_counts[speaker] = 0
                appear_counts[speaker] += 1

        # Compute ratio
        win_rates = {}
        for speaker, appear_ct in appear_counts.items():
            w_ct = win_counts.get(speaker, 0)
            win_rates[speaker] = w_ct / appear_ct

        return win_rates
