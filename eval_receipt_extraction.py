import os
import requests
import asyncio
from dotenv import load_dotenv
from httpx import AsyncClient
from braintrust import EvalAsync
from autoevals import Factuality, Levenshtein

from receipt_extraction import Deps, extraction_agent

load_dotenv()
RAW_DATA_URL = os.getenv('RAW_DATA_URL')
NUM_RECEIPTS = 1
def load_receipt(index):
    json_path = f"{RAW_DATA_URL}/key/{index}.json"
    json_response = requests.get(json_path).json()
    return json_response

async def braintrust_evaluation(client):
    indices = [str(i).zfill(3) for i in range(NUM_RECEIPTS)]
    bulk_data = [
        {
            "input": {
                "key": key,
                "idx": idx,
            },
            "expected": value,
        }

        for idx, (fields) in [
            (idx, load_receipt(idx)) for idx in indices[:NUM_RECEIPTS]
        ]
        for key, value in fields.items()
    ]
    
    async def task(input):
        deps = Deps(client=client, index=input["idx"], key=input["key"])
        return await extraction_agent.run(
            f'What is the {deps.key} in receipt index {deps.index}?', deps=deps
        )
    
    await EvalAsync(
        "Receipt Extraction",
        data=bulk_data,
        task=task,
        scores=[Levenshtein, Factuality],
        experiment_name=f"Receipt Extraction - 'gpt-4o'",
        metadata={"model": 'gpt-4o'},
    )

async def main():
    async with AsyncClient() as client:
        await braintrust_evaluation(client)


if __name__ == '__main__':
    asyncio.run(main())