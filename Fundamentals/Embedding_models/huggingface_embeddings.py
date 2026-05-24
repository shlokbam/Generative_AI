from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model = 'sentence-transformers/all-MiniLM-L6-v2',
)

docs = ["hello this is shlok", "cricket is a good game"]
vector = embedding.embed_documents(docs)
print(vector)   