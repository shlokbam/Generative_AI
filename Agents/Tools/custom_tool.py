from langchain.tools import tool

@tool
def get_greetings(name: str) -> str:
    """This function is used to generate greetings for the user"""
    return f"Hello {name}, Welcome to AI World"

result = get_greetings.invoke({"name": "Shlok"})

print(result)
print(get_greetings.name)
print(get_greetings.description)
print(get_greetings.args)