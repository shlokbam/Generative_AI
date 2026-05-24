import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.1,
)

file_path = "/Users/shlokbam/Documents/Code/Generative_AI/RAG_Project/Big.pdf"

data = PyPDFLoader(file_path)

docs = data.load()


splitted_docs = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitted_docs.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI that summarizs the text"),
        ("human", "{data}")
    ]
)

prompt = template.format_messages(data=docs)

response = llm.invoke(prompt)
print(response.content)
