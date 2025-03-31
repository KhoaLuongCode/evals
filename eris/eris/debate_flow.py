from dataclasses import dataclass
from typing import List, Tuple
from llm_interface import LLMInterface


@dataclass
class DebateTopic:
    """
    Holds the text and positions for a given debate topic.
    """
    text: str
    pro_position: str = "Pro"
    con_position: str = "Con"


@dataclass
class DebateRound:
    """
    Stores an individual round's speaker and content.
    """
    speaker: str
    content: str


def debate_transcript(
    pro_llm: LLMInterface,
    con_llm: LLMInterface,
    topic: DebateTopic,
    max_tokens: int = 256,
    temperature: float = 0.7
) -> List[DebateRound]:
    """
    Orchestrates a structured debate and returns a list of DebateRound objects.
    This uses a functional style: one function that runs the entire debate flow.

    Debate structure:
      1. Pro Constructive
      2. Con Constructive
      3. Pro Cross-Ex
      4. Con Cross-Ex
      5. Pro Rebuttal
      6. Con Rebuttal
      7. Pro Closing
      8. Con Closing
    """
    rounds: List[DebateRound] = []

    pro_constructive_prompt = (
        f"You are '{topic.pro_position}'. Present your constructive argument. "
        f"Topic: {topic.text}"
    )
    con_constructive_prompt = (
        f"You are '{topic.con_position}'. Present your constructive argument. "
        f"Topic: {topic.text}"
    )

    cross_ex_prompt = (
        "Now cross-examine your opponent's argument. Be direct and critical."
        "\nOpponent's statement:\n\"\"\"{opponent_statement}\"\"\""
    )
    rebuttal_prompt = (
        "Rebut your opponent's cross-examination or argument."
        "\nOpponent's statement:\n\"\"\"{opponent_statement}\"\"\""
    )
    closing_prompt = "Provide a concise closing argument summarizing why your position is correct."

    # 1. Pro Constructive
    pc = pro_llm.generate_response(pro_constructive_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=pro_llm.name, content=pc))

    # 2. Con Constructive
    cc = con_llm.generate_response(con_constructive_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=con_llm.name, content=cc))

    # 3. Pro Cross-Ex
    px_prompt = cross_ex_prompt.format(opponent_statement=cc)
    px = pro_llm.generate_response(px_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=pro_llm.name, content=px))

    # 4. Con Cross-Ex
    cx_prompt = cross_ex_prompt.format(opponent_statement=pc)
    cx = con_llm.generate_response(cx_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=con_llm.name, content=cx))

    # 5. Pro Rebuttal
    pr_prompt = rebuttal_prompt.format(opponent_statement=cx)
    pr = pro_llm.generate_response(pr_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=pro_llm.name, content=pr))

    # 6. Con Rebuttal
    cr_prompt = rebuttal_prompt.format(opponent_statement=px)
    cr = con_llm.generate_response(cr_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=con_llm.name, content=cr))

    # 7. Pro Closing
    p_close = pro_llm.generate_response(closing_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=pro_llm.name, content=p_close))

    # 8. Con Closing
    c_close = con_llm.generate_response(closing_prompt, max_tokens, temperature)
    rounds.append(DebateRound(speaker=con_llm.name, content=c_close))

    return rounds
