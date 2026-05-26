from dotenv import load_dotenv
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langgraph.prebuilt import create_react_agent

# wrap_tool_call doesn't exist in LangChain — simple pass-through decorator
wrap_tool_call = lambda fn: fn

load_dotenv()

#weather tool
@tool
def get_weather( city: str ) -> str:
    """This function is used to get the weather of the city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city}: {desc}, {temp}°C"

#news tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
@tool
def get_news(city : str) -> str:
    """Get latest news about a city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])
    if not results:
        return "No news found"
    
    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


print(get_news.invoke({"city": "Pune"}))


llm = ChatMistralAI(model_name="mistral-small")

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

agent = create_react_agent(
    llm,
    tools=[get_weather,get_news],
    prompt="You are a helpful assistant."

)

print("City Agent | type exit to quit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print("Agent:", result["messages"][-1].content)
