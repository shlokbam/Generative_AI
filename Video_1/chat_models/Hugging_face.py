import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="conversational",
    max_new_tokens=200,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)
Response = model.invoke("What is cricket?")
print(Response.content)