import asyncio
import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import redis

# Load environment variables
load_dotenv()

# MCP + LangChain
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

def get_session_key(uid: str) -> str:
    return f"session:{uid}"

# FastAPI models
class QueryRequest(BaseModel):
    query: str
    uid: Optional[str] = None

class QueryResponse(BaseModel):
    uid: str
    response: str
    is_new_session: bool

# Grafana MCP Client
class GrafanaMCPClient:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    async def query_with_history(self, chat_history: List[Dict[str, str]]) -> str:
        server_params = StdioServerParameters(
            command="docker",
            args=[
                "run", "--rm", "-i", "--network=host",
                "-e", "GRAFANA_URL",
                "-e", "GRAFANA_API_KEY",
                "mcp/grafana",
                "-t", "stdio"
            ],
            env={
                "GRAFANA_URL": os.environ["GRAFANA_URL"],
                "GRAFANA_API_KEY": os.environ["GRAFANA_API_KEY"]
            }
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(self.model, tools)
                result = await agent.ainvoke({"messages": chat_history})
                return result["messages"][-1].content

# FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Grafana MCP API is running"}

@app.post("/query")
async def query_grafana(request: QueryRequest):
    client = GrafanaMCPClient()

    # UID logic
    if request.uid is None:
        uid = str(uuid.uuid4())
        is_new_session = True
        redis_client.hset(f"meta:{uid}", mapping={
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        })
    else:
        uid = request.uid
        is_new_session = False
        if not redis_client.exists(get_session_key(uid)):
            redis_client.hset(f"meta:{uid}", mapping={
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat()
            })
            is_new_session = True
        else:
            redis_client.hset(f"meta:{uid}", "last_used", datetime.now().isoformat())

    # Load full chat history
    history_raw = redis_client.lrange(get_session_key(uid), 0, -1)
    chat_history = []
    for item in history_raw:
        entry = json.loads(item)
        chat_history.append({"role": "user", "content": entry["query"]})
        chat_history.append({"role": "assistant", "content": entry["response"]})
    # Add current query
    chat_history.append({"role": "user", "content": request.query})

    try:
        response_text = await client.query_with_history(chat_history)

        # Save new interaction
        entry = {
            "query": request.query,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        redis_client.rpush(get_session_key(uid), json.dumps(entry))

        return QueryResponse(
            uid=uid,
            response=response_text,
            is_new_session=is_new_session
        )
    except Exception as e:
        return {"error": str(e), "uid": uid}

@app.get("/session/{uid}")
async def get_session_history(uid: str):
    if not redis_client.exists(get_session_key(uid)):
        return {"error": "Session not found"}

    history_raw = redis_client.lrange(get_session_key(uid), 0, -1)
    history = [json.loads(item) for item in history_raw]
    meta = redis_client.hgetall(f"meta:{uid}")

    return {
        "uid": uid,
        "history": history,
        "created_at": meta.get("created_at"),
        "last_used": meta.get("last_used"),
        "total_queries": len(history)
    }

@app.get("/sessions")
async def list_all_sessions():
    session_keys = redis_client.keys("meta:*")
    sessions = []

    for key in session_keys:
        uid = key.split(":")[1]
        meta = redis_client.hgetall(key)
        total_queries = redis_client.llen(get_session_key(uid))
        sessions.append({
            "uid": uid,
            "created_at": meta.get("created_at"),
            "last_used": meta.get("last_used"),
            "total_queries": total_queries
        })

    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

# Optional CLI test
async def main():
    client = GrafanaMCPClient()
    history = [{"role": "user", "content": "Query 1"}]
    response = await client.query_with_history(history)
    print(response)

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Test MCP")
    print("2. Run FastAPI")
    choice = input("Enter choice (1 or 2): ")
    if choice == "1":
        asyncio.run(main())
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
