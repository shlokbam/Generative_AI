import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(os.path.dirname(current_dir), "document_loaders", "GRU.pdf")

data = PyPDFLoader(file_path)

docs = data.load()

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

splitted_text = splitter.split_documents(docs)

print(splitted_text[0].page_content)