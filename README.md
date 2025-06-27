# RAG Chatbot (LangChain + Streamlit)

This project is a production-style **Retrieval-Augmented Generation (RAG)** chatbot built with:

- **Streamlit** (UI)
- **LangChain** (RAG logic)
- **Google Gemini** (LLM + Embeddings)
- **Chroma** (Vector store)

It lets users **chat with their own PDFs**, maintains **conversation memory**, and supports **multiple conversations** — similar to ChatGPT's chat history.

---

## ✨ Features

- ✅ Upload and index multiple PDF documents
- ✅ Ask questions grounded in uploaded docs (RAG)
- ✅ General-purpose chat even without any docs
- ✅ Conversational memory per session
- ✅ Multiple named conversations (like ChatGPT chats)
- ✅ Delete individual documents from the index
- ✅ Clean, user-friendly Streamlit interface

---

## 📸 Demo Screenshot

![Aditya_Borhade-Day 10-Progress](https://github.com/user-attachments/assets/fe401c13-a934-47d0-bad4-ad7b4d7f9edc)

---

## 🚀 How It Works

> The app implements **Retrieval-Augmented Generation (RAG)** as follows:

1. PDFs are split into text chunks
2. Embeddings are generated using Google's model
3. Chroma indexes them for semantic search
4. User questions retrieve relevant chunks
5. The LLM generates grounded answers

✅ If no PDFs are uploaded, it falls back to normal assistant chat.

---

## SetUp Instructions

1️⃣ **Clone the Repository**

`git clone <your-repo-url>`
`cd <your-repo>`

2️⃣ **Install Dependencies**

`pip install -r requirements.txt`

3️⃣ **Set Up Environment Variables(.env)**

`GOOGLE_API_KEY=your_google_api_key_here`

4️⃣ Run the App Locally

`streamlit run src/day10.py`

---

## 🗨️ Usage Guide

- Upload PDFs in the sidebar
- Process PDFs to build/update the semantic index
- See your uploaded documents and delete any single one if needed
- Start multiple conversations, each with its own memory
- Chat in the main window — with context from uploaded docs
- Fallback to general-purpose assistant if no docs are uploaded
