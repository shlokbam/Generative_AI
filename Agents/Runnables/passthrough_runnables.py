from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(model_name="mistral-small")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a code generator"),
        ("human", "{topic}")
    ]
)

explain_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who explains code in simple terms"),
        ("human", "Explain the following code in simple words:\n{code}")
    ]
)

seq1 = code_prompt | model | parser

seq2 = RunnableParallel(
    code = RunnablePassthrough(),
    explanation = explain_prompt | model | parser
)

final_seq = seq1 | seq2

response = final_seq.invoke({"topic": "Write a code of palindrome in java"})

print("Code:", response['code'])
print("\n" + "="*40 + "\n")
print("Explanation:", response['explanation'])
