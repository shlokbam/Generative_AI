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

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI that summarizs the text"),
        ("human", "{data}")
    ]
)

prompt = template.format_messages(data="What is the capital of France?")

response = llm.invoke(prompt)
print(response.content)
