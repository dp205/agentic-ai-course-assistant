from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key="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"
)

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
import chromadb
import datetime

# -------------------------------
# 1. STATE (MANDATORY FIRST STEP)
# -------------------------------
class CapstoneState(TypedDict):
    question: str
    messages: List[str]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int


# -------------------------------
# 2. SETUP (Reuse KB)
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="course_assistant")


# -------------------------------
# 3. NODES
# -------------------------------

# MEMORY NODE
def memory_node(state: CapstoneState):
    messages = state.get("messages", [])
    messages.append("USER: " + state["question"])

    # sliding window (last 6 messages)
    messages = messages[-6:]

    return {"messages": messages}


# ROUTER NODE
def router_node(state: CapstoneState):
    question = state["question"]

    prompt = f"""
You are a router for an AI assistant.

Decide the route for the user query.

Options:
- retrieve → if question needs knowledge base
- tool → if question needs calculation, time, or external info
- skip → if greeting or casual

Respond with ONLY one word: retrieve / tool / skip

Question: {question}
"""

    route = llm.invoke(prompt).content.strip().lower()

    return {"route": route}


# RETRIEVAL NODE
def retrieval_node(state: CapstoneState):
    query = state["question"]
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = ""
    sources = []

    for i in range(len(docs)):
        context += f"[{metas[i]['topic']}]\n{docs[i]}\n\n"
        sources.append(metas[i]["topic"])

    return {"retrieved": context, "sources": sources}


# SKIP NODE
def skip_node(state: CapstoneState):
    return {"retrieved": "", "sources": []}


# TOOL NODE
def tool_node(state: CapstoneState):
    question = state["question"].lower()

    try:
        if "time" in question:
            result = str(datetime.datetime.now())

        elif "multiply" in question:
            nums = [int(s) for s in question.split() if s.isdigit()]
            result = str(nums[0] * nums[1])

        else:
            result = "Tool not applicable"

    except Exception:
        result = "Error in tool"

    return {"tool_result": result}


# ANSWER NODE
def answer_node(state: CapstoneState):

    context = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages = "\n".join(state.get("messages", []))

    prompt = f"""
You are an AI Course Assistant.

STRICT RULES:
- Answer ONLY using provided context
- If answer not in context, say "I don't know"
- Do NOT hallucinate

Conversation:
{messages}

Context:
{context}

Tool Result:
{tool_result}

Question:
{state["question"]}

Answer clearly:
"""

    response = llm.invoke(prompt).content

    return {"answer": response}


# EVAL NODE
def eval_node(state: CapstoneState):

    if not state.get("retrieved"):
        return {
            "faithfulness": 1.0,
            "eval_retries": state.get("eval_retries", 0) + 1
        }

    prompt = f"""
Check if the answer is based ONLY on the context.

Context:
{state['retrieved']}

Answer:
{state['answer']}

Give a score between 0.0 and 1.0
Only output number.
"""

    score_text = llm.invoke(prompt).content.strip()

    try:
        score = float(score_text)
    except:
        score = 0.5

    retries = state.get("eval_retries", 0)

    return {
        "faithfulness": score,
        "eval_retries": retries + 1
    }

# SAVE NODE
def save_node(state: CapstoneState):
    messages = state.get("messages", [])
    messages.append("ASSISTANT: " + state["answer"])

    return {"messages": messages}


# -------------------------------
# 4. GRAPH BUILDING
# -------------------------------

def route_decision(state: CapstoneState):
    return state["route"]


def eval_decision(state: CapstoneState):
    if state["faithfulness"] < 0.7 and state["eval_retries"] < 2:
        return "answer"
    return "save"


graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")

graph.add_edge("memory", "router")

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "retrieve": "retrieve",
        "skip": "skip",
        "tool": "tool"
    }
)

graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")

graph.add_edge("answer", "eval")

graph.add_conditional_edges(
    "eval",
    eval_decision,
    {
        "answer": "answer",
        "save": "save"
    }
)

graph.add_edge("save", END)

app = graph.compile(checkpointer=MemorySaver())

print("✅ Graph compiled successfully")


# -------------------------------
# 5. TEST FUNCTION
# -------------------------------
def ask(question, thread_id="1"):
    result = app.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id}}
    )
    return result["answer"]


# -------------------------------
# 6. RUN TEST
# -------------------------------
if __name__ == "__main__":
    print(ask("What is LangGraph?"))
    print(ask("What did I just ask?", thread_id="1"))