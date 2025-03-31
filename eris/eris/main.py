import os
import json
import logging

from llm_interface import OpenAIInterface, AnthropicInterface
from debate_flow import DebateTopic
from judge_module import DebateResult
from aggregator import Aggregator
from config import OPENAI_DEFAULT, ANTHROPIC_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_eris():
    """
    Entry point to set up Eris debates and run them.
    """
    # 1) Create LLMs for debate participants
    openai_llm_1 = OpenAIInterface(
        name="OpenAI-GPT4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", OPENAI_DEFAULT.api_key),
        model=OPENAI_DEFAULT.model
    )

    openai_llm_2 = OpenAIInterface(
        name="OpenAI-GPT4o",
        api_key=os.getenv("OPENAI_API_KEY", OPENAI_DEFAULT.api_key),
        model="gpt-4o"
    )

    anthropic_llm_1 = AnthropicInterface(
        name="Anthropic-Claude",
        api_key=os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_DEFAULT.api_key),
        model=ANTHROPIC_DEFAULT.model
    )

    # 2) Create judge LLM(s)
    #    Could be a single or list [OpenAIInterface(...), AnthropicInterface(...)]
    judge = [
        # Example: We'll use an ensemble of GPT3.5 and Claude
        openai_llm_2,
        anthropic_llm_1
    ]

    # 3) Create topics
    topics = [
        DebateTopic(text="Should universal basic income be implemented globally?"),
    ]

    # 4) List out LLM pairs to debate
    llm_pairs = [
        [openai_llm_1, openai_llm_2],    
        [openai_llm_2, anthropic_llm_1]  
    ]

    # 5) Create aggregator, run debates, collect results
    aggregator = Aggregator(topics=topics, judge_llms=judge)
    aggregator.run_debates(llm_pairs, num_debates=2, max_tokens=256, temperature=0.7)
    
    # 6) Compute and display stats
    win_rates = aggregator.compute_win_rates()
    logging.info("=== Win Rates ===")
    for speaker, rate in win_rates.items():
        logging.info(f"{speaker}: {rate:.2%}")

    results_path = os.path.join(os.path.dirname(__file__), "eris_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        # Convert each DebateResult into a dict
        serialized = []
        for r in aggregator.results:
            serialized.append({
                "topic": r.topic,
                "winner": r.winner,
                "judge_reasoning": r.judge_reasoning,
                "transcript": [
                    {"speaker": d.speaker, "content": d.content} for d in r.transcript
                ]
            })
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    logging.info(f"Debate results saved to: {results_path}")


if __name__ == "__main__":
    run_eris()
