"""
Web search tool for the builder agent — uses Serper API for Google search + httpx for page scraping.

Gives the LLM real-world data (brand info, features, pricing, copy) before generating code,
so landing pages and company sites use actual content instead of placeholders.
"""

import os
import httpx
from langchain_core.tools import tool

# Domains that block scraping or return useless content
_BLOCKED_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "linkedin.com", "reddit.com", "youtube.com", "tiktok.com",
    "pinterest.com", "yelp.com",
}

_SERPER_URL = "https://google.serper.dev/search"


async def _serper_search(query: str, num_results: int = 5) -> dict:
    """Call Serper API for Google search results."""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return {"error": "SERPER_API_KEY not configured"}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            _SERPER_URL,
            json={"q": query, "num": num_results},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


async def _scrape_page(url: str) -> str:
    """Scrape a page and return text content (best-effort, max 5000 chars)."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Buildable/1.0)"})
            resp.raise_for_status()

            # Extract text from HTML (simple approach — strip tags)
            import re
            html = resp.text
            # Remove script and style blocks
            html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text[:5000] if len(text) > 100 else ""
    except Exception as e:
        print(f"Scrape failed for {url}: {e}")
        return ""


async def _run_web_search(query: str, max_results: int = 5) -> str:
    """Execute web search and return formatted results for the LLM."""
    try:
        data = await _serper_search(query, max_results)
    except Exception as e:
        return f"Web search failed: {str(e)}"

    if "error" in data:
        return f"Web search error: {data['error']}"

    parts: list[str] = []

    # Knowledge graph (company info)
    kg = data.get("knowledgeGraph", {})
    if kg:
        parts.append("=== KNOWLEDGE GRAPH ===")
        if kg.get("title"):
            parts.append(f"Name: {kg['title']}")
        if kg.get("description"):
            parts.append(f"Description: {kg['description']}")
        if kg.get("website"):
            parts.append(f"Website: {kg['website']}")
        attrs = kg.get("attributes", {})
        for key, val in list(attrs.items())[:6]:
            parts.append(f"{key}: {val}")
        parts.append("")

    # Answer box
    ab = data.get("answerBox", {})
    if ab:
        parts.append("=== DIRECT ANSWER ===")
        if ab.get("title"):
            parts.append(f"Title: {ab['title']}")
        if ab.get("answer"):
            parts.append(f"Answer: {ab['answer']}")
        if ab.get("snippet"):
            parts.append(f"Snippet: {ab['snippet']}")
        parts.append("")

    # Organic results (snippets)
    organic = data.get("organic", [])[:5]
    if organic:
        parts.append("=== SEARCH RESULTS ===")
        for i, result in enumerate(organic, 1):
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            parts.append(f"{i}. {title}")
            parts.append(f"   URL: {link}")
            parts.append(f"   {snippet}")
            parts.append("")

    # Scrape top 2 non-blocked pages for full content
    scrape_urls = []
    for result in organic:
        url = result.get("link", "")
        if url and not any(blocked in url for blocked in _BLOCKED_DOMAINS):
            scrape_urls.append(url)
        if len(scrape_urls) >= 2:
            break

    if scrape_urls:
        import asyncio
        scraped = await asyncio.gather(*[_scrape_page(url) for url in scrape_urls])
        for url, content in zip(scrape_urls, scraped):
            if content:
                parts.append(f"=== PAGE CONTENT: {url} ===")
                parts.append(content)
                parts.append("")

    return "\n".join(parts) if parts else "No results found."


def create_web_search_tool():
    """Create the web_search tool for LangChain agents."""

    @tool
    async def web_search(query: str) -> str:
        """Search the web for real-world information. Use this BEFORE generating code for
        landing pages, company pages, or any topic-specific content.

        Search for: company info, features, pricing, testimonials, brand colors.
        Example queries:
        - "Stripe official website features pricing"
        - "Airbnb landing page design 2024"
        - "Tesla products pricing specifications"

        NEVER use vague single-word queries — always add descriptive qualifiers."""
        return await _run_web_search(query, max_results=5)

    return web_search
