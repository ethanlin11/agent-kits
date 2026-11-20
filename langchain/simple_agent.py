# pip install -qU langchain "langchain[openai]"
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent

env_path = Path(__file__).parent.parent / ".env"

load_dotenv(dotenv_path=env_path)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}"


agent = create_agent(
    model="openai:auto",  # 模型提供商标志:模型名称
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

print(result["messages"][-1].content_blocks)
