import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

llm_gemini_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
llm_gemini_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
llm_claude_sonnet = (
    ChatAnthropic(model="claude-sonnet-4-5", anthropic_api_key=anthropic_key, max_tokens=8096)
    if anthropic_key
    else None
)
llm_claude_haiku = (
    ChatAnthropic(model="claude-haiku-4-5-20251001", anthropic_api_key=anthropic_key, max_tokens=4096)
    if anthropic_key
    else None
)
