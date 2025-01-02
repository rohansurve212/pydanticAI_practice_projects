from __future__ import annotations as _annotations

import asyncio
import os
from io import BytesIO
import fitz  # PyMuPDF
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
from utils import input_with_default

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure()

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
RAW_DATA_URL = os.getenv('RAW_DATA_URL')
model = OpenAIModel('gpt-4o', api_key=OPENAI_API_KEY)
DOCUMENT_TYPE = ""

@dataclass
class Deps_Receipt:
    client: AsyncClient
    index: str
    key: str

@dataclass
class Deps_Contract:
    client: AsyncClient
    file_path: str
    key: str

class Receipt(BaseModel):
    company: str = Field(description='The name of the company or the business that issued the receipt')
    date: str = Field(description='The date of issue on the receipt')
    address: str = Field(description='The address of the company or the business that issued the receipt')
    total: str = Field(description='The total bill amount on the receipt')

class Contract(BaseModel):
    licensor: str = Field(description="The party in a licensing agreement that grants another party (the licensee) the right to produce, use, sell, and/or display the licensor’s protected material")
    licensee: str = Field(description="The party in a licensing agreement that receives from another party (the licensor) the right to produce, use, sell, and/or display the licensor’s protected material")
    termination_clause: str = Field(description="The Termination clause is a crucial aspect of licensing agreements, as they outline the circumstances under which the agreement can be ended")
    signing_date: str = Field(description="The date of signing of the licensing contract between the two parties")

# Create the Receipt Extractor Agent
receipt_extraction_agent = Agent(
    model,
    deps_type=Deps_Receipt,
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

# Create the Contract Extractor Agent
contract_extraction_agent = Agent(
    model,
    deps_type=Deps_Contract,
    result_type=Contract,
    system_prompt=(
        "Use 'extract_text_from_pdf' tool to extract from the PDF document. "
        "Return the requested information from the extracted text content of the PDF. "
        "Structure the output according to the Contract class definition. "
        "If any field cannot be identified, return it as null. Do not include explanations or markdown, only return the structured output."
    ),
    retries=2,
)

# Tool to fetch and process the image
@receipt_extraction_agent.tool
async def process_receipt_image(ctx: RunContext[Deps_Receipt]) -> str:
    with logfire.span('processing receipt image using OCR') as span:
        image_url = f"{RAW_DATA_URL}/img/{ctx.deps.index}.jpg"
        response = await ctx.deps.client.get(image_url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        text = image_to_string(image)  # Using Tesseract OCR to process image to text 
        span.set_attribute('response', text)
    return text

# Tool to extract text from the PDF
@contract_extraction_agent.tool
def extract_text_from_pdf(ctx: RunContext[Deps_Contract]) -> str:
    with logfire.span('extracting text from PDF using PyMuPDF') as span:
        file_path = ctx.deps.file_path
        extracted_text = ""

        with fitz.open(file_path) as pdf: # Using PyMuPDF to extract text from PDF
            for page in pdf:
                extracted_text += page.get_text()
    return extracted_text

# Tool for Receipt Extractor Agent to fetch the key field dynamically
@receipt_extraction_agent.tool
def fetch_key_field(ctx: RunContext[Deps_Receipt]) -> str:
    return ctx.deps.key

# Tool for Contract Extractor Agent to fetch the key field dynamically
@contract_extraction_agent.tool
def fetch_key_field(ctx: RunContext[Deps_Contract]) -> str:
    return ctx.deps.key

async def main():
    DOCUMENT_TYPE = input_with_default("Please provide the document type - Receipt OR Contract?: ", "Receipt")
    async with AsyncClient() as client:
        if DOCUMENT_TYPE == "Receipt":
            deps_receipt = Deps_Receipt(client=client, index='069', key='company')
            result = await receipt_extraction_agent.run(
                f'What is the {deps_receipt.key} in receipt index {deps_receipt.index}?', deps=deps_receipt
            )
        else:
            file_path = os.path.join("ChinaRealEstateInformationCorp_20090929_1.pdf")
            absolute_path = os.path.abspath(file_path)
            deps_contract = Deps_Contract(client=client, file_path=f'{absolute_path}', key='licensor')
            result = await contract_extraction_agent.run(
                f'What is the {deps_contract.key} in the contract?', deps=deps_contract
            )
        print('Response:', result.data)

if __name__ == '__main__':
    asyncio.run(main())