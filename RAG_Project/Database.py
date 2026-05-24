import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

file_path = "/Users/shlokbam/Documents/Code/Generative_AI/RAG_Project/Big.pdf"

data = PyPDFLoader(file_path)

docs = data.load()


splitted_docs = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitted_docs.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed")

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "chroma_db")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)





