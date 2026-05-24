from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings = MistralAIEmbeddings(model="mistral-embed")

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(os.path.dirname(current_dir), "chroma_db")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_dir
)

result = vectorstore.similarity_search("What is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

