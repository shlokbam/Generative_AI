from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

search_tool = TavilySearchResults(max_results=5)
llm = ChatMistralAI(model_name="mistral-small")
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)

chain = prompt| llm | parser

news_result = search_tool.invoke("Latest AI news of 2026")

response = chain.invoke({"news": news_result})

print(response)


print(search_tool.description)
print(search_tool.name)
print(search_tool.args)