import streamlit as st
from rag_core import retrieval_chain
st.title("RAG for textbooks")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
user_question = st.text_input("Ask a question:")
if st.button("Submit") and user_question:
    response = retrieval_chain.invoke({"input": user_question})
    answer = response["answer"]
    st.write(answer)
for chat in st.session_state.chat_history:
    st.write(chat["question"])
    st.write(chat["answer"])