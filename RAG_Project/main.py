import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "chroma_db")

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm = ChatMistralAI(
    model="mistral-small-latest"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

            Use ONLY the provided context to answer the question.

            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
