import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

llm_gemini_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
llm_gemini_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)