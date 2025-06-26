import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import shutil
import gc
import time
import uuid

load_dotenv()
persist_dir = f"chroma_db_{uuid.uuid4().hex[:6]}"
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)


def process_pdfs(pdfs):
    raw_text = ""
    filenames = []

    for pdf in pdfs:
        filenames.append(pdf.name)
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(raw_text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    try:
        if "vectorstore" in st.session_state:
            st.session_state.vectorstore = None
        existing = get_vectorstore()
        if existing:
            if hasattr(existing, "__del__"):
                existing.__del__()
            del existing
        gc.collect()
        time.sleep(1)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
    except Exception as e:
        new_name = f"{persist_dir}_{uuid.uuid4().hex[:6]}"
        os.rename(persist_dir, new_name)
        st.warning(f"⚠️ Old vectorstore renamed to `{new_name}` due to lock: {e}")

    vectorstore = Chroma.from_documents(documents, embedding=embedding, persist_directory=persist_dir)
    st.session_state.vectorstore = vectorstore
    st.session_state.uploaded_files = filenames
    st.success("✅ PDFs processed and embedded!")
    return vectorstore


def get_vectorstore():
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)
    return None


def get_qa_chain(vstore):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question based on the following context. Be concise and accurate.

Context:
{context}

Question:
{question}

Answer:
"""
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )


st.set_page_config("📄 Chat with Multiple PDFs", layout="centered")
st.title("📄 Chat with Multiple PDFs (Gemini + Chroma)")

with st.sidebar:
    st.header("Upload PDFs")
    uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("📄 Process PDFs") and uploaded_pdfs:
        try:
            process_pdfs(uploaded_pdfs)
        except Exception as e:
            st.error(f"❌ Failed to process files: {e}")

    if st.session_state.get("uploaded_files"):
        st.markdown("**📂 Uploaded Docs:**")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- `{fname}`")

vstore = st.session_state.get("vectorstore") or get_vectorstore()

if vstore:
    qa_chain = get_qa_chain(vstore)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("💬 Ask a question from your documents:")

    if st.button("Send") and user_query:
        answer = qa_chain.run(user_query)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("AI", answer))

    for speaker, text in st.session_state.chat_history:
        st.markdown(f"**{'🧑‍💻' if speaker == 'You' else '🤖'} {speaker}:** {text}")
else:
    st.info("👆 Upload and process PDFs to begin.")
