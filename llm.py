import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
import dotenv

# Load API key
dotenv.load_dotenv(dotenv_path="config/.env")
api_key = os.getenv("OPENAI_API_KEY")

# Set model
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

# Sending messages
response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Dieter"),
        AIMessage(content="Hello Felix! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
print(f"ChatGPT response: {response.content}")
