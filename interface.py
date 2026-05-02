import streamlit as st
import tempfile
from rag_core import build_rag_pipeline
st.title("RAG for Textbooks")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
uploaded_file = st.file_uploader("Upload your textbook PDF", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    st.session_state.retrieval_chain = build_rag_pipeline(temp_path)
    st.success("Document uploaded successfully.")
if st.session_state.retrieval_chain:
    user_question = st.text_input("Ask a question:")
    if st.button("Submit") and user_question:
        response = st.session_state.retrieval_chain.invoke({"input": user_question})
        answer = response["answer"]
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
for chat in st.session_state.chat_history:
    st.write(chat["question"])
    st.write(chat["answer"])