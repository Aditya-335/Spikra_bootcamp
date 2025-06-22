from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os, shutil

def run_day5():
    load_dotenv()

    docs = [
        Document(page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.", metadata={"team": "Royal Challengers Bangalore"}),
        Document(page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.", metadata={"team": "Mumbai Indians"}),
        Document(page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.", metadata={"team": "Chennai Super Kings"}),
        Document(page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.", metadata={"team": "Mumbai Indians"}),
        Document(page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.", metadata={"team": "Chennai Super Kings"}),
    ]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_dir = "chroma_db"

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    query = input("\n Enter your cricket-related question: ")

    results = vectorstore.similarity_search_with_score(query, k=2)

    print("\n Top Matching Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] {doc.page_content.strip()}")
        print(f"   Team: {doc.metadata.get('team')}")
        print(f"   Similarity Score: {score:.4f}")
