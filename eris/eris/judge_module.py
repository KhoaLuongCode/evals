from typing import List, Union
from dataclasses import dataclass
from llm_interface import LLMInterface
from debate_flow import DebateRound, DebateTopic

@dataclass
class DebateResult:
    """
    Stores the final outcome of a judged debate, including
    the entire transcript, the judge's reasoning, and the winner.
    """
    topic: str
    winner: str
    judge_reasoning: str
    transcript: List[DebateRound]

def judge_debate(
    judge_llms: Union[LLMInterface, List[LLMInterface]],
    topic: DebateTopic,
    transcript: List[DebateRound],
    max_tokens: int = 512,
    temperature: float = 0.7
) -> DebateResult:
    """
    Evaluates the debate with one or multiple judges.
    Returns a DebateResult with the winner and judge reasoning.
    """

    if isinstance(judge_llms, list):
        # Ensemble approach
        results = []
        for jllm in judge_llms:
            r = _single_judge(jllm, topic, transcript, max_tokens, temperature)
            results.append(r)

        # Use majority vote or fallback
        winners = [res.winner for res in results]
        final_winner = max(set(winners), key=winners.count)
        combined_reasoning = "\n\n".join([f"[{res.winner} - {res.judge_reasoning}]" for res in results])

        return DebateResult(
            topic=topic.text,
            winner=final_winner,
            judge_reasoning=f"Ensemble Decision:\n{combined_reasoning}",
            transcript=transcript
        )
    else:
        # Single LLM judge
        return _single_judge(judge_llms, topic, transcript, max_tokens, temperature)


def _single_judge(
    judge_llm: LLMInterface,
    topic: DebateTopic,
    transcript: List[DebateRound],
    max_tokens: int,
    temperature: float
) -> DebateResult:
    """
    Runs a single judge LLM on the transcript.
    """
    # Format transcript
    transcript_text = "\n".join(
        [f"{r.speaker}: {r.content}" for r in transcript]
    )

    # Construct the judge prompt
    judge_prompt = f"""
You are a neutral debate judge. The debate topic is: "{topic.text}".
Below is the full transcript of the debate:
\"\"\"
{transcript_text}
\"\"\"

Evaluate which side made the stronger arguments. 
Consider:
- Argument strength
- Logical consistency
- Evidence use
- Rebuttal quality
- Overall persuasiveness

Provide a short explanation, then declare the winner by name.

Response Format:
Reasoning: ...
Winner: ...
    """

    response = judge_llm.generate_response(judge_prompt, max_tokens, temperature)

    # Handle both string and list[TextBlock]
    if isinstance(response, list):
        response = "\n".join([block.text if hasattr(block, "text") else str(block) for block in response])

    extracted_winner = _parse_winner(response, [r.speaker for r in transcript])

    

    return DebateResult(
        topic=topic.text,
        winner=extracted_winner,
        judge_reasoning=response,
        transcript=transcript
    )


def _parse_winner(response: str, speaker_names: List[str]) -> str:
    """
    Naive extraction: looks for 'Winner:' line or checks speaker names.
    More robust approach could use regex or a structured response.
    """
    lower_resp = response.lower()
    if "winner:" in lower_resp:
        lines = lower_resp.splitlines()
        for line in lines:
            if "winner:" in line.lower():
                # Attempt to match a known speaker
                for s in speaker_names:
                    if s.lower() in line.lower():
                        return s
                # If no match found, just strip
                return line.split(":", 1)[-1].strip()

    # Fallback: guess by frequency
    winners_count = {s: lower_resp.count(s.lower()) for s in speaker_names}
    # Return the speaker whose name appears the most
    return max(winners_count, key=winners_count.get)
