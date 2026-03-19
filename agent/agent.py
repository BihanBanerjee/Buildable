from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Builder models the user can choose from
MODELS = {
    "gemini-pro": "google/gemini-2.5-pro",
    "claude-sonnet": "anthropic/claude-sonnet-4",
}

# Maps each builder model to its lightweight sibling for the fast nodes
FAST_MODEL_MAP = {
    "google/gemini-2.5-pro": "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4": "anthropic/claude-haiku-4.5",
}

DEFAULT_BUILDER_MODEL = MODELS["gemini-pro"]


def get_fast_model(builder_model: str) -> str:
    """Return the lightweight model that matches the builder's provider family."""
    return FAST_MODEL_MAP.get(builder_model, "google/gemini-2.5-flash")


def create_llm(api_key: str, model: str, max_tokens: int = 8096) -> ChatOpenAI:
    """Create a ChatOpenAI instance routed through OpenRouter."""
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        max_tokens=max_tokens,
    )
