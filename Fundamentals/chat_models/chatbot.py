import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

messages = [
    SystemMessage(content= "You are a funny ai agent"),
    
]   

while True:
    print("="*100)
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "exit":
        break
    Response = llm.invoke(messages)
    print("AI: ",Response.content)
    messages.append(AIMessage(content=Response.content))
    print("="*100)

print(messages)