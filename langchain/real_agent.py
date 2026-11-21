from pathlib import Path
from dotenv import load_dotenv
import urllib.error
import urllib.request
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from deepagents import create_deep_agent

env_path = Path(__file__).parent.parent / ".env"

load_dotenv(dotenv_path=env_path)


SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions--ground them in tool results from the saved file."""


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL"""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text


model = init_chat_model(
    "openai:auto",
    # temperature=0.5,
    timeout=300,
    max_tokens=25000,
)

checkpointer = InMemorySaver()

# LangChain agents
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# deep agents
deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = f"""Standard Ebooks hosts a full HTML copy of Beatrix Potter's short story "The Tale of Ginger and Pickles".
URL: https://standardebooks.org/ebooks/beatrix-potter/short-fiction/text/the-tale-of-ginger-and-pickles

Answer as much as you can:

1) How many lines in the fetched text contain the substring `Ginger` (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the fetched text that contains `Pickles`.
3) A two-sentence neutral synopsis of the story.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""

print("Running create_agent...", flush=True)
agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "the-tale-of-ginger-and-pickles-lc"}},
)
print("\ncreate_agent:")
print(agent_result["messages"][-1].content_blocks)

print("Running create_deep_agent...", flush=True)
deep_agent_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "the-tale-of-ginger-and-pickles-da"}},
)
print("\ncreate_deep_agent:")
print(deep_agent_result["messages"][-1].content_blocks)
