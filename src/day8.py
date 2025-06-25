import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
persist_dir = "chroma_db"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def process_pdfs(pdfs):
    raw_text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    docs = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = Chroma.from_documents(
        docs,
        embedding=embedding,
        persist_directory=persist_dir
    )

    st.success("✅ PDF processed and vectorstore updated!")
    return vectorstore


def get_vectorstore():
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)
    return None

def get_qa_chain(vstore):
    retriever = vstore.as_retriever()
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question based on the following context. Be precise and do not make up anything.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

st.set_page_config(page_title="📄 PDF Chatbot", layout="centered")
st.title("📄 Chat with Your PDF (Gemini + Chroma)")

with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_pdfs = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs") and uploaded_pdfs:
        process_pdfs(uploaded_pdfs)

vstore = get_vectorstore()
if vstore:
    qa_chain = get_qa_chain(vstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("💬 Ask a question from your PDFs:")

    if st.button("Send") and query:
        answer = qa_chain.run(query)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("AI", answer))

    for speaker, text in st.session_state.chat_history:
        st.markdown(f"**{'🧑‍💻' if speaker == 'You' else '🤖'} {speaker}:** {text}")
else:
    st.warning("👆 Upload and process a PDF to begin chatting.")
