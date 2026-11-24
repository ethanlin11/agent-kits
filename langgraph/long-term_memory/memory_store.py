import asyncio
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langchain_core.utils.uuid import uuid7
from langchain.chat_models import init_chat_model

env_path = Path(__file__).parent.parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

model = init_chat_model("openai:auto")

store = InMemoryStore()


@dataclass
class Context:
    user_id: str


async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memory")

    # Search for relevant memories
    memories = await runtime.store.asearch(
        namespace, query=state["messages"][-1].content, limit=3
    )
    info = "\n".join([d.value["data"] for d in memories])
    print(f"Get info from long-term memories: {info}")

    # ... Use memories as a part of prompts in model call
    response = await model.ainvoke(
        [{"role": "system", "content": f"User-specific data as follows: {info}"}]
        + state["messages"]
    )
    print(f"AI answer: {response.content}")

    # Store a new memory
    await runtime.store.aput(
        namespace, str(uuid7()), {"data": "Jerry likes playing basketball."}
    )


# Insert simulated data into the store
store.put(("Jerry", "memory"), str(uuid7()), {"data": "Jerry comes from China."})

builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(store=store)

# Pass context at invocation time
if __name__ == "__main__":
    asyncio.run(
        graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "My name is Jerry. I like playing basketball. Do you know where I come from?",
                    }
                ]
            },
            context=Context(user_id="Jerry"),
        )
    )
