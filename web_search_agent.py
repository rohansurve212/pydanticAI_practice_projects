from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import logfire
from devtools import debug
from httpx import AsyncClient
from dotenv import load_dotenv

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
LLM = os.getenv('LLM_MODEL', 'gpt-4o')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY')
model = OpenAIModel(LLM, api_key=OPENAI_API_KEY)

logfire.configure()

@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

# Create the Agent
web_search_agent = Agent(
    model,
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        f'You are an expert at researching the web to answer user questions. '
        'The current date is: {datetime.now().strftime("%Y-%m-%d")}'
    ),
    deps_type=Deps,
    retries=2,
)

# Create a Function Tool for the agent
@web_search_agent.tool
async def search_web(
    ctx: RunContext[Deps], web_query: str
) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."
    
    headers = {
        'X-Subscription-Token': ctx.deps.brave_api_key,
        'Accept': 'application/json',
    }
    
    params = {
        'q': web_query,
        'count': 5,
        'text_decorations': True,
        'search_lang': 'en'
    }
    
    with logfire.span('calling Brave search API', params=params, headers=headers) as span:
        print(f"WEB QUERY --> {web_query}")
        r = await ctx.deps.client.get(
            'https://api.search.brave.com/res/v1/web/search',
            params=params,
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    results = []
    
    # Add web results in a nice formatted way
    web_results = data.get('web', {}).get('results', [])
    for item in web_results[:3]:
        title = item.get('title', '')
        description = item.get('description', '')
        url = item.get('url', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."

async def main():
    async with AsyncClient() as client:
        brave_api_key = BRAVE_API_KEY
        deps = Deps(client=client, brave_api_key=brave_api_key)
        result = await web_search_agent.run(
            'Give me some articles talking about the benefits of using the Rust programming language.', deps=deps
        )
        debug(result)
        print('Response:', result.data)

if __name__ == '__main__':
    asyncio.run(main())