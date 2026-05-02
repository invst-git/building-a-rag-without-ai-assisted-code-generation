import streamlit as st
from rag_core import retriever_chain
st.title("RAG for textbooks")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
user_question = st.text_input("Ask a question:")
if st.button("Submut") and user_question:
    response = retriever_chain.invoke({"question":user_question, "answer":answer})
for chat in st.session_state.chat_history:
    st.write(chat["question"])
    st.write(chat["answer"])