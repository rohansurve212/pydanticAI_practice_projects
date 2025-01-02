from __future__ import annotations as _annotations

import asyncio
import os
from io import BytesIO
from dotenv import load_dotenv
from dataclasses import dataclass
from PIL import Image
from pytesseract import image_to_string
from typing import Any, List, Tuple

import logfire
from devtools import debug
from httpx import AsyncClient

from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure()

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
RAW_DATA_URL = os.getenv('RAW_DATA_URL')
model = OpenAIModel('gpt-4o', api_key=OPENAI_API_KEY)

@dataclass
class Deps:
    client: AsyncClient
    index: str
    key: str

class Receipt(BaseModel):
    company: str = Field(description='The name of the company or the business that issued the receipt')
    date: str = Field(description='The date of issue on the receipt')
    address: str = Field(description='The address of the company or the business that issued the receipt')
    total: str = Field(description='The total bill amount on the receipt')

# Create the Receipt Extractor Agent
extraction_agent = Agent(
    model,
    deps_type=Deps,
    result_type=Receipt,
    system_prompt=(
        "Use 'process_receipt_image' to process receipt image. "
        "Extract the requested field from the OCR-processed text of the image. "
        "Return the extracted value, and nothing else. "
        "For example, if the field is 'total' and the value is '100', "
        "you should just return '100'. If the field is not present, return null. "
        "Do not decorate the output with any explanation, or markdown. Just return the extracted value."
    ),  
    retries=2,
)

# Tool to fetch and process the image
@extraction_agent.tool
async def process_receipt_image(ctx: RunContext[Deps]) -> str:
    with logfire.span('processing receipt image using OCR') as span:
        image_url = f"{RAW_DATA_URL}/img/{ctx.deps.index}.jpg"
        response = await ctx.deps.client.get(image_url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        text = image_to_string(image)  # OCR processing using Tesseract
        span.set_attribute('response', text)
    return text

# Tool to fetch the key field dynamically
@extraction_agent.tool
def fetch_key_field(ctx: RunContext[Deps]) -> str:
    return ctx.deps.key

async def main():
    async with AsyncClient() as client:
        deps = Deps(client=client, index='069', key='company')
        result = await extraction_agent.run(
            f'What is the {deps.key} in receipt index {deps.index}?', deps=deps
        )
        print('Response:', result.data)

if __name__ == '__main__':
    asyncio.run(main())