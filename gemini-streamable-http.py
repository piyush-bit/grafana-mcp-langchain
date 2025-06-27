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
from datetime import datetime
from zoneinfo import ZoneInfo
import tzlocal
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# MCP + LangChain with HTTP Streamable support
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

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

# Grafana MCP Client with HTTP Streamable Transport
class GrafanaMCPClient:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
        # HTTP Streamable server URL (your MCP server endpoint)
        self.grafana_server_url = os.environ.get("GRAFANA_MCP_SERVER_URL", "http://localhost:8000")

    async def query_with_history(self, chat_history: List[Dict[str, str]]) -> str:
        # Get system's local timezone
        local_tz = ZoneInfo(tzlocal.get_localzone_name())
        current_time = datetime.now(local_tz)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        system_prompt = """
You are an AI assistant embedded as a plugin inside Grafana. You support users through a chat interface with access to the Grafana Machine Control Plugin (MCP) and a full suite of observability tools. Your role is to help users monitor, investigate, and improve their system health using Grafana.

You have access to dashboards, Prometheus, Loki, alerting, incidents, OnCall, Sift, and Pyroscope tools. You can retrieve data, run queries, analyze logs and metrics, and update or create resources (dashboards, alerts, incidents, etc.).

Always act with clarity and decisiveness. Instead of asking vague or open-ended questions, suggest **direct actions or multiple choices** the user can pick from. Aim to reduce cognitive load and move users quickly toward results.

Examples of good behavior:
- "I found 3 dashboards matching 'CPU'. Would you like to: (1) View them (2) Run a Prometheus query (3) Add a panel?"
- "The alert rule is firing. Do you want to: (1) View logs (2) Create an incident (3) Silence the alert?"
- "Here are 2 slow queries found in the last hour. Should I: (1) Show related logs (2) Create a panel (3) Start a Sift investigation?"

✅ Capabilities
- Search, create, and update dashboards and panels.
- Run PromQL and LogQL queries and summarize results.
- Analyze logs, find error patterns, detect anomalies.
- Create, update, and resolve incidents.
- View and manage alert rules and OnCall schedules.
- Investigate performance issues with Pyroscope and Sift.

🧠 Behavior
- Be proactive. Suggest concrete actions or next steps.
- Use numbered options when possible, to make replies scannable and actionable.
- If user input is ambiguous, infer likely intent and provide 2–3 best options to continue.
- Prefer doing and showing, rather than asking.

🛑 Do Not:
- Ask broad or vague questions like "What would you like to do?" or "Can you clarify?"
- Assume the user has deep technical knowledge—explain briefly if needed.
- Invent resources or metrics—use only real data from tools.

Your mission is to reduce friction, suggest next best actions, and guide users confidently through Grafana to solve problems or gain insights fast.
"""



        # System prompt with dynamic date/time
        system_prompt = {
            "role": "system",
            "content": (
                f"You are a helpful assistant having access to grafana your job is to assist the user. Current system date and time is {formatted_time}. "
                "Provide accurate and concise responses."
            )
        }
        chat_history = [system_prompt] + chat_history
        # Configure HTTP Streamable server
        client = MultiServerMCPClient(
            {
                "grafana": {
                    "url": f"{self.grafana_server_url}/mcp",
                    "transport": "streamable_http",
                }
            }
        )

        # Get all tools from the Grafana MCP server
        tools = await client.get_tools()
        
        # Create agent with the tools
        agent = create_react_agent(self.model, tools)
        
        # Execute with chat history
        result = await agent.ainvoke({"messages": chat_history})
        return result["messages"][-1].content

    # Alternative method using specific server tools
    async def query_with_server_specific_tools(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Get tools from specific server only
        """
        client = MultiServerMCPClient(
            {
                "grafana": {
                    "url": f"{self.grafana_server_url}/mcp",
                    "transport": "streamable_http",
                }
            }
        )
        
        # Get tools from specific server
        tools = await client.get_tools(server_name="grafana")
        
        # Create agent
        agent = create_react_agent(self.model, tools)
        
        # Execute with chat history
        result = await agent.ainvoke({"messages": chat_history})
        return result["messages"][-1].content

# FastAPI app (rest remains the same)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # <- allowed origins
    allow_credentials=True,
    allow_methods=["*"],            # <- allow all HTTP methods
    allow_headers=["*"],            # <- allow all headers
)

@app.get("/")
async def root():
    return {"message": "Grafana MCP API with HTTP Streamable Transport is running"}

@app.post("/query")
async def query_grafana(request: QueryRequest):
    client = GrafanaMCPClient()

    # UID logic (same as before)
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

    # Load full chat history (same as before)
    history_raw = redis_client.lrange(get_session_key(uid), 0, -1)
    chat_history = []
    for item in history_raw:
        entry = json.loads(item)
        chat_history.append({"role": "user", "content": entry["query"]})
        chat_history.append({"role": "assistant", "content": entry["response"]})
    
    # Add current query
    chat_history.append({"role": "user", "content": request.query})

    try:
        # Use HTTP Streamable transport
        response_text = await client.query_with_history(chat_history)
        # or use: response_text = await client.query_with_server_specific_tools(chat_history)

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

# Rest of the endpoints remain the same
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

# Test function for HTTP Streamable
async def main():
    client = GrafanaMCPClient()
    history = [{"role": "user", "content": "list available grafana tools"}]
    response = await client.query_with_history(history)
    print(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)