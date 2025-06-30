from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline, Conversation
import nltk
import sqlite3
from datetime import datetime
import uvicorn
import logging
import os

# Download necessary NLTK resources
nltk.download('punkt')

# Initialize FastAPI app
app = FastAPI(title="AI-Powered Chatbot", description="A contextual chatbot for customer support.", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load conversational model
try:
    qa_pipeline = pipeline("conversational", model="microsoft/DialoGPT-small")
    logger.info("Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Model loading failed.")

# SQLite setup
DB_NAME = "chat_logs.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()

# Request schema
class Message(BaseModel):
    text: str

# In-memory conversation object
conversation = Conversation()

@app.post("/chat", summary="Interact with the chatbot", response_description="Bot's response")
async def chat(message: Message):
    try:
        global conversation
        conversation.add_user_input(message.text)
        result = qa_pipeline(conversation)
        bot_response = result.generated_responses[-1]

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs (user_input, bot_response, timestamp)
                VALUES (?, ?, ?)
            ''', (message.text, bot_response, datetime.now().isoformat()))
            conn.commit()

        return {"response": bot_response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/logs", summary="Get recent chat logs", response_description="List of chat interactions")
def get_logs(limit: int = 10):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_input, bot_response, timestamp FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
            logs = cursor.fetchall()
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Log retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
