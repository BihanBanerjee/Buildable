"""
LLM configuration — hardcoded models via OpenRouter (BYOK).

Build agent: anthropic/claude-sonnet-4.5 (high quality initial generation)
Edit agent:  openai/o4-mini (fast follow-up edits)
"""

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BUILD_MODEL = "anthropic/claude-sonnet-4.5"
EDIT_MODEL = "openai/o4-mini"


def create_build_llm(api_key: str) -> ChatOpenAI:
    """Create LLM for initial builds (Sonnet 4.5)."""
    return ChatOpenAI(
        model=BUILD_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
    )


def create_edit_llm(api_key: str) -> ChatOpenAI:
    """Create LLM for edits and error fixes (o4-mini)."""
    return ChatOpenAI(
        model=EDIT_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
    )
