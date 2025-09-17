import os
import numpy as np
from typing import Union, Dict
from memory import retrieve_relevant_chunks
from embedding import embed_text
from groq import Groq
from dotenv import load_dotenv
from tools import calculate, fetch_news  # your utility tools

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Store messages per session
session_messages: Dict[str, list] = {}

tools = [
    {"name": "calculator", "desc": "Can perform arithmetic calculations, parse natural language math queries."},
    {"name": "pdf_summarizer", "desc": "Can summarize PDF documents and provide key points from text."},
    {"name": "news_fetcher", "desc": "Can fetch latest news articles about a topic."}
]

# Initialize embeddings for tools
def init_tools():
    for tool in tools:
        if "embedding" not in tool:
            tool["embedding"] = embed_text(tool["desc"])
    return tools
def extract_pdf_topics(pdf_path, max_chunks=10):
    chunks = retrieve_relevant_chunks(pdf_path, top_k=50)[:max_chunks]
    topics = []

    for chunk in chunks:
        topic_text = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Read the following PDF content and suggest 5 concise, meaningful topic titles in 3-5 words:\n{chunk}"
            }]
        ).choices[0].message.content.strip()

        # Extract numbered list (1., 2., 3., …)
        for line in topic_text.split("\n"):
            line = line.strip()
            if line and any(char.isalpha() for char in line):
                # Remove leading number and dot
                line = line.split(". ", 1)[-1]
                topics.append(line)

    # Deduplicate while preserving order
    unique_topics = list(dict.fromkeys(topics))
    return unique_topics[:10]  # top 10 topics


# --- PDF Summarization ---
def summarize_topic(pdf_path, topic, max_chunks=10):
    chunks = retrieve_relevant_chunks(pdf_path, top_k=50)[:max_chunks]
    relevant_chunks = [c for c in chunks if topic.lower() in c.lower()]
    detailed_summary = []
    for chunk in relevant_chunks:
        summary = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Summarize this chunk:\n{chunk}"}]
        ).choices[0].message.content
        detailed_summary.append(summary)
    return "\n\n".join(detailed_summary)

# --- Cosine similarity ---
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# --- Main agent dispatch ---
def agentic_dispatch(user_input: Union[str, dict], session_id: str = "default"):
    if session_id not in session_messages:
        session_messages[session_id] = [{"role": "system", "content": "You are a helpful AI assistant."}]
    messages = session_messages[session_id]
    outputs = []

    if isinstance(user_input, str) and user_input.lower() in ["thanks", "thank you", "thx"]:
        return "You're welcome!"

    active_tools = init_tools()
    query_embedding = embed_text(user_input if isinstance(user_input, str) else "")
    tool_scores = [(tool["name"], cosine_similarity(query_embedding, tool["embedding"])) for tool in active_tools]
    tool_scores.sort(key=lambda x: x[1], reverse=True)
    selected_tools = [name for name, score in tool_scores if score > 0.5]

    # PDF
    pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]
    pdf_path = os.path.join("uploads", pdf_files[-1]) if pdf_files else None

    if "pdf_summarizer" in selected_tools and isinstance(user_input, str):
        if pdf_path:
            topics = extract_pdf_topics(pdf_path)
            return {"topics": topics} if topics else {"error": "No meaningful topics found."}
        return {"error": "No PDF uploaded."}

    if isinstance(user_input, dict) and "pdf_topic" in user_input:
        if pdf_path:
            summary = summarize_topic(pdf_path, user_input["pdf_topic"])
            return {"summary": summary}
        return {"error": "No PDF uploaded."}

    # News
    tool_contexts = []
    if "news_fetcher" in selected_tools and isinstance(user_input, str):
        news_text = fetch_news(user_input, page_size=5)
        tool_contexts.append(f"News:\n{news_text}")

    # Calculator
    if "calculator" in selected_tools and isinstance(user_input, str):
        calc_result = calculate(user_input)
        if calc_result is not None:
            tool_contexts.append(f"Calculator Result:\n{calc_result}")

    user_text = user_input if isinstance(user_input, str) else ""
    if tool_contexts:
        combined_context = "\n\n".join(tool_contexts)
        messages.append({"role": "user", "content": combined_context + "\n\n" + user_text})
    else:
        messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
    ai_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": ai_reply})
    session_messages[session_id] = messages
    outputs.append(f"AI:\n{ai_reply}")

    return "\n\n".join(outputs)
