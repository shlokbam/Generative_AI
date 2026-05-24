import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.1,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "document_loaders", "GRU.pdf")

data = PyPDFLoader(file_path)

docs = data.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI that summarizs the text"),
        ("human", "{data}")
    ]
)

prompt = template.format_messages(data=docs[-1].page_content)

response = llm.invoke(prompt)
print(response.content)
