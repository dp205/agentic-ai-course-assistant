# capstone_streamlit.py

import streamlit as st
from agent import app  # importing your LangGraph app
import uuid

# 1. PAGE CONFIG
st.set_page_config(page_title="Course Assistant", layout="wide")

st.title("🤖 Agentic AI Course Assistant")

# 2. SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# 3. SIDEBAR

with st.sidebar:
    st.header("About")
    st.write("AI Assistant for Agentic AI Course")
    
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 4. DISPLAY CHAT

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# 5. USER INPUT

user_input = st.chat_input("Ask something")

if user_input:
    # show user message
    st.chat_message("user").write(user_input)

    # store message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 6. CALL AGENT

    with st.spinner("Thinking..."):
        result = app.invoke(
            {"question": user_input},
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )

        answer = result["answer"]

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })