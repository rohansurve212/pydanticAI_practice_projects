import os
from typing import Tuple
import requests
import asyncio
from dotenv import load_dotenv
from httpx import AsyncClient
from rapidfuzz.distance import Levenshtein

from document_extraction import Deps_Receipt, Receipt, receipt_extraction_agent
from eval_receipt_extraction import load_receipt

load_dotenv()
RAW_DATA_URL = os.getenv('RAW_DATA_URL')
NUM_RECEIPTS = 3

def evaluate(model_answer: Receipt, reference_answer: Receipt) -> Tuple[float, str]:
    score = 0
    reason = []
    errors = {}
    errors['company'] = Levenshtein.distance(model_answer.company, reference_answer['company'])
    errors['date'] = Levenshtein.distance(model_answer.date, reference_answer['date'])
    errors['address'] = Levenshtein.distance(model_answer.address, reference_answer['address'])
    errors['total'] = Levenshtein.distance(model_answer.total, reference_answer['total'])

    # if errors['company'] in answer.company:
    #     score += 0.25
    #     reason.append("Correct company identified")
    # if reference_answer.date in answer.date:
    #     score += 0.25
    #     reason.append("Correct date identified")
    # height_error = abs(reference_answer.height - answer.height)
    # if height_error < 10:
    #     score += 0.25 * (10 - height_error)/10.0
    #     reason.append(f"Height was {height_error}m off. Correct answer is {reference_answer.height}")
    # else:
    #     reason.append(f"Wrong mountain identified. Correct answer is {reference_answer.name}")

    # return score, ';'.join(reason)
    return errors

async def custom_evaluation(client):
    indices = [str(i).zfill(3) for i in range(NUM_RECEIPTS)]
    
    model_answers = []
    reference_answers = []
    
    for idx in indices[:NUM_RECEIPTS]:
        data = load_receipt(idx)
        reference_answers.append(data)
        
        deps_receipt = Deps_Receipt(client=client, index=idx, key="company")
        result = await receipt_extraction_agent.run(
            f'What is the {deps_receipt.key} in receipt index {deps_receipt.index}?', deps=deps_receipt
        )
        model_answers.append(result.data)
        print(f'Response for receipt {idx}:', result.data)
    
    for model_answer, reference_answer in zip(model_answers, reference_answers):
        errors = evaluate(model_answer, reference_answer)
        print(errors)

async def main():
    async with AsyncClient() as client:
        await custom_evaluation(client)


if __name__ == '__main__':
    asyncio.run(main())