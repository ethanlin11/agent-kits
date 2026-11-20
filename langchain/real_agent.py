from pathlib import Path
from dotenv import load_dotenv
import urllib.error
import urllib.request
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

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
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
)

checkpoint = InMemorySaver()
