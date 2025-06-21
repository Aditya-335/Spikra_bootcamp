from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import re

def init_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        temperature=0.7,
        google_api_key=api_key
    )

def qna_mode(llm):
    question = input("\n Enter your question: ")

    template = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful AI assistant. Answer the following question:\n\nQuestion: {question}"
    )
    prompt = template.format(question=question)
    response = llm.invoke(prompt)
    print("\n Answer:\n", clean_output(response.content))

def summarize_mode(llm):
    text = input("\n Paste the text you want to summarize:\n")

    template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in 3-4 sentences:\n\n{text}"
    )
    prompt = template.format(text=text)
    response = llm.invoke(prompt)
    print("\n Answer:\n", clean_output(response.content))

def clean_output(text: str) -> str:
    text = re.sub(r'[*_`#>]', '', text)

    text = text.replace('\\n', '\n')

    text = text.replace("\\'", "'").replace('\\"', '"')

    
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()

def run_day3():
    llm = init_gemini()
    while True:
        print("\n--- Chatbot Modes ---")
        print("1. Q&A")
        print("2. Summarization")
        print("3. Exit")
        choice = input("Choose a mode (1/2/3): ").strip()

        if choice == "1":
            qna_mode(llm)
        elif choice == "2":
            summarize_mode(llm)
        elif choice == "3":
            print("Exiting chatbot. Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")
