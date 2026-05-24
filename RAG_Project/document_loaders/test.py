import os
from langchain_community.document_loaders import TextLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "notes.txt")

data = TextLoader(file_path)

documents = data.load()
print(documents[0].page_content)