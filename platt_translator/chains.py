"""LangChain chain definitions for translation and validation."""

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_PROMPT_DIR = Path(__file__).parent


def _load_prompt(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")


def build_translator_chain(llm):
    """Build a chain that translates High German → Francique rhénan lorrain.

    The system message combines SYSTEM_PROMPT.md and DIALECT_RULES.md.
    The human message is the German text to translate.
    """
    system_prompt = _load_prompt("SYSTEM_PROMPT.md")
    dialect_rules = _load_prompt("DIALECT_RULES.md")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{system_prompt}\n\n{dialect_rules}"),
        ("human", "{text}"),
    ])
    return prompt | llm | StrOutputParser()


def build_validator_chain(llm):
    """Build a chain that validates a Francique rhénan lorrain sentence.

    Returns one of: VALID, QUESTIONABLE, INVALID.
    """
    validator_template = _load_prompt("VALIDATOR_PROMPT.md")
    # Replace the {{TEXT}} placeholder with LangChain's {text} variable
    validator_template = validator_template.replace("{{TEXT}}", "{text}")

    prompt = ChatPromptTemplate.from_messages([
        ("human", validator_template),
    ])
    return prompt | llm | StrOutputParser()
