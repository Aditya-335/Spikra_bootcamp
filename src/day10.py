import streamlit as st
import uuid
import os
import shutil
import gc
import time

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

BASE_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.4



def build_vectorstore(conv_data):
    try:
        if os.path.exists(conv_data["persist_dir"]):
            shutil.rmtree(conv_data["persist_dir"])
            time.sleep(0.1)
            gc.collect()
    except Exception as e:
        st.warning(f"⚠️ Error clearing old vectorstore: {e}")

    if not conv_data["uploaded_docs"]:
        return

    all_docs = []
    for name, text in conv_data["uploaded_docs"]:
        chunks = splitter.split_text(text)
        all_docs.extend([Document(page_content=chunk, metadata={"source": name}) for chunk in chunks])

    Chroma.from_documents(
        all_docs,
        embedding=embedding,
        persist_directory=conv_data["persist_dir"]
    )

st.set_page_config(page_title="🧠 RAG Chatbot", layout="wide")
st.title("🤖 Welcome to RAGify")

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}  

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None

embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

with st.sidebar:
    st.header("💼 Conversations")

    # Create new
    new_conv_name = st.text_input("🆕 New conversation name")
    if st.button("Create Conversation"):
        if not new_conv_name.strip():
            st.warning("Please enter a valid name.")
        elif new_conv_name in st.session_state.conversations:
            st.warning("Conversation already exists!")
        else:
            session_id = uuid.uuid4().hex
            persist_dir = f"{BASE_PERSIST_DIR}_{uuid.uuid4().hex[:8]}"
            st.session_state.conversations[new_conv_name] = {
                "session_id": session_id,
                "chat_history": [],
                "uploaded_docs": [],
                "persist_dir": persist_dir,
            }
            st.session_state.current_conversation = new_conv_name
            st.success(f"✅ Created conversation '{new_conv_name}'")

    if st.session_state.conversations:
        all_convs = list(st.session_state.conversations.keys())
        selected = st.selectbox(
            "🔀 Switch conversation",
            all_convs,
            index=all_convs.index(st.session_state.current_conversation) if st.session_state.current_conversation in all_convs else 0
        )
        st.session_state.current_conversation = selected

        st.markdown(f"✅ **Current:** `{selected}`")

        if st.button("🗑️ Delete This Conversation"):
            conv = st.session_state.conversations.pop(selected, None)
            if conv and os.path.exists(conv["persist_dir"]):
                try:
                    shutil.rmtree(conv["persist_dir"])
                except Exception as e:
                    st.warning(f"⚠️ Error deleting vectorstore: {e}")
            st.session_state.current_conversation = None
            st.rerun()

    else:
        st.info("Create a new conversation to begin.")

if not st.session_state.current_conversation:
    st.stop()

conv_data = st.session_state.conversations[st.session_state.current_conversation]

st.sidebar.markdown("---")
st.sidebar.subheader("📂 PDF Management")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if st.sidebar.button("➕ Add PDFs") and uploaded_files:
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
        conv_data["uploaded_docs"].append((pdf.name, text))
    st.sidebar.success("✅ PDFs added!")
    build_vectorstore(conv_data)


if conv_data["uploaded_docs"]:
    st.sidebar.subheader("📜 Uploaded Documents")
    for idx, (name, _) in enumerate(conv_data["uploaded_docs"]):
        col1, col2 = st.sidebar.columns([4, 1])
        col1.markdown(f"- {name}")
        if col2.button("❌", key=f"del_{idx}"):
            conv_data["uploaded_docs"].pop(idx)
            build_vectorstore(conv_data)
            st.rerun()



st.markdown("---")
st.subheader(f"💬 Chat in: **{st.session_state.current_conversation}**")

user_input = st.text_input("Your message:", key="input_box", value="")

if st.button("Send"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        conv_data["chat_history"].append(("You", user_input))

        if conv_data["uploaded_docs"] and os.path.exists(conv_data["persist_dir"]):
            # RAG flow
            try:
                retriever = Chroma(
                    persist_directory=conv_data["persist_dir"],
                    embedding_function=embedding
                ).as_retriever()

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever,
                    return_source_documents=False
                )
                result = qa_chain(
                    {"question": user_input, "chat_history": conv_data["chat_history"]}
                )
                answer = result["answer"]
            except Exception as e:
                answer = f"⚠️ Retrieval error: {e}"
        else:
            
            try:
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                for u, a in conv_data["chat_history"]:
                    messages.append({"role": "user", "content": u})
                    messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user", "content": user_input})
                answer = llm.invoke(messages).content
            except Exception as e:
                answer = f"⚠️ LLM error: {e}"

        conv_data["chat_history"].append(("AI", answer))
        st.rerun()

for speaker, text in conv_data["chat_history"]:
    if speaker == "You":
        st.markdown(f"**🧑‍💻 You:** {text}")
    elif speaker == "AI":
        st.markdown(f"**🤖 AI:** {text}")
