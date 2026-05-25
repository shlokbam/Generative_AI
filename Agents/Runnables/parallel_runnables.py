from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableLambda

load_dotenv()

model = ChatMistralAI(model_name="mistral-small")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {short} in short words."
)

long_prompt = ChatPromptTemplate.from_template(
    "Explain {detailed} in detail."
)

# chain = RunnableParallel(
#     short=RunnableLambda(lambda x: {'short': x['short']}) | short_prompt | model | parser,
#     detailed=RunnableLambda(lambda x: {'detailed': x['detailed']}) | long_prompt | model | parser
# )

chain = RunnableParallel(
    short = short_prompt | model | parser,
    detailed= long_prompt | model | parser
)

response = chain.invoke(
    {'short': 'Machine Learning', 'detailed': 'Deep Learning'}
)

print(response['short'])
print(response['detailed'])
