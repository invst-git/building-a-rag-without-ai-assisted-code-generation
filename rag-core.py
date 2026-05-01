from langchin_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectors import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_document_chain
# loading
loader = PyPDFLoader("Documents\Analysis1.pdf")
documents = loader.load()
# chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=256)
chunks = splitter.split_documents(documents)
# embeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
# using FAISS to keep the vectorDB simple for this implementation
vectordb = FAISS.from_documents(documents=chunks,embeddings=embeddings)
#retriever
retriever = vectordb.as_retriever()
