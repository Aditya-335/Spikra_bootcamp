from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os, shutil

def run_day6():
    load_dotenv()

    docs = [
        Document(page_content="Mercury is the closest planet to the Sun and has a very thin atmosphere.", metadata={"planet": "Mercury"}),
        Document(page_content="Venus is the second planet from the Sun. It has a thick, toxic atmosphere that traps heat.", metadata={"planet": "Venus"}),
        Document(page_content="Earth is the only known planet to support life, with vast oceans and a breathable atmosphere.", metadata={"planet": "Earth"}),
        Document(page_content="Mars is known as the Red Planet due to iron oxide on its surface. It has two moons: Phobos and Deimos.", metadata={"planet": "Mars"}),
        Document(page_content="Jupiter is the largest planet in the solar system and is famous for its Great Red Spot.", metadata={"planet": "Jupiter"}),
    ]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_dir = "chroma_day6"

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = input("\n Ask a space-related question: ")

    response = qa_chain.invoke({"query": query})

    print("\n Answer:")
    print(response["result"])

    print("\n Retrieved Context:")
    for doc in response["source_documents"]:
        print("-", doc.page_content.strip())
        print(f"   Planet: {doc.metadata.get('planet')}\n")
