from google import genai
from dotenv import load_dotenv
import os

def run_day1():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Who is Elon Musk?"
    )
    print("Day 1 Output:", response.text)
