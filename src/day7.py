from langgraph.graph import END, StateGraph
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

import os

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

memory_store = {}

def get_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

def respond(state: dict) -> dict:
    history = get_memory(state["config"]["session_id"])
    user_message = HumanMessage(content=state["input"])

    history.add_message(user_message)

    messages = history.messages
    response = llm.invoke(messages)

    history.add_message(response)

    return {
        "messages": messages + [user_message, response],
        "config": state["config"],
    }

class ChatState(TypedDict):
    input: str
    config: dict
    messages: list

graph = StateGraph(ChatState)
graph.add_node("respond", respond)
graph.set_entry_point("respond")
graph.add_edge("respond", END)

chatbot = graph.compile()

def run_day7():
    print(" LangGraph + Gemini chatbot with memory. Type 'exit' to quit.")
    session_id = "user-123"

    while True:
        user_input = input(" You: ")
        if user_input.lower() == "exit":
            break

        result = chatbot.invoke({
            "input": user_input,
            "config": {"session_id": session_id}
        })

        ai_response = result["messages"][-1]
        print(" AI:", ai_response.content)
