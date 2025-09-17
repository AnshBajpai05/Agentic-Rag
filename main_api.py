from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from agent import agentic_dispatch, extract_pdf_topics, summarize_topic
import os
from memory import store_pdf

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "FastAPI backend is running"}

# --- Upload PDF ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        from pathlib import Path
        safe_name = Path(file.filename).name
        file_path = os.path.join(UPLOAD_FOLDER, safe_name)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        store_pdf(file_path)
        return {"reply": f"PDF uploaded successfully: {file.filename}"}
    except Exception as e:
        return {"error": str(e)}

# --- Chat endpoint ---
@app.post("/chat")
async def chat_endpoint(payload: dict):
    user_input = payload.get("user_input", "")
    try:
        reply = agentic_dispatch(user_input, session_id="default")
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}

# --- PDF topics ---
@app.get("/pdf/topics")
async def pdf_topics():
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        return {"error": "No PDF uploaded"}
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_files[-1])
    try:
        topics = extract_pdf_topics(pdf_path)
        return {"topics": topics}
    except Exception as e:
        return {"error": str(e)}

# --- PDF summary ---
@app.post("/pdf/summary")
async def pdf_summary(payload: dict):
    topic = payload.get("topic", "")
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        return {"error": "No PDF uploaded"}
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_files[-1])
    try:
        summary = summarize_topic(pdf_path, topic)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/chat")
async def chat_endpoint(payload: dict):
    user_input = payload.get("user_input", "")
    try:
        reply = agentic_dispatch(user_input, session_id="default")

        # If reply is dict (PDF topics/summary), return directly
        if isinstance(reply, dict):
            return reply
        
        # Otherwise, wrap as plain reply
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}

