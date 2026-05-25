from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)

llm = ChatMistralAI(
    model_name="mistral-small",
    temperature=0.2
)

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({"topic": "Machine Learning"})

print(response)