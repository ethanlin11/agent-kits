from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

env_path = Path(__file__).parent.parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

model = init_chat_model("openai:auto")

DB_URI = "postgresql://myuser:mypassword@localhost:5432/mydb?sslmode=disable"


async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        async def call_model(state: MessagesState):
            response = await model.ainvoke(state["messages"])
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "2"}}

        stream = await graph.astream_events(
            {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
            config,
            version="v3",
        )

        async for message in stream.messages:
            async for token in message.text:
                print(token, end="", flush=True)

        stream = await graph.astream_events(
            {"messages": [{"role": "user", "content": "what's my name?"}]},
            config,
            version="v3",
        )

        async for message in stream.messages:
            async for token in message.text:
                print(token, end="", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
