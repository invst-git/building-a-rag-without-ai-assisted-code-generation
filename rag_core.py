from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
def build_rag_pipeline(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectordb.as_retriever()
    prompt = PromptTemplate(input_variables=["context", "input"], template="""You are a helpful assistant for answering natural language queries from the loaded document. You must strictly confine your responses to the loaded document. Use only the provided context to answer the question. If the answer is not found in the context, respond with: "Answer not found!" Context: {context} Question: {input} Answer:""")
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, anthropic_api_key=api_key)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain