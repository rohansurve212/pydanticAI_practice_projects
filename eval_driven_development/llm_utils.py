from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models import Model
from pydantic_ai import Agent
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
model = GeminiModel('gemini-1.5-flash', api_key=GEMINI_API_KEY)

def default_model() -> Model:
    model = GeminiModel('gemini-1.5-flash', api_key=GEMINI_API_KEY)
    return model

def agent() -> Agent:
    return Agent(default_model())