import os
from langchain_community.document_loaders import PyPDFLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "GRU.pdf")

data = PyPDFLoader(file_path)

docs = data.load()

print(docs[-1].page_content)