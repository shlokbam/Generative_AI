# from gitdb.fun import chunk_size
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(os.path.dirname(current_dir), "document_loaders", "notes.txt")

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1,
)

data = TextLoader(file_path)
documents = data.load()

chunks = splitter.split_documents(documents)

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()