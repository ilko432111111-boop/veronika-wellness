from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = supabase.table("documents").select("content").execute()
    context = "\n".join([item["content"] for item in response.data]) if response.data else "No information available."
    
    prompt = f"""You are a helpful assistant for Veronika's wellness business.
Use only the information below to answer.

Information:
{context}

Customer: {request.message}
Answer:"""

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {"reply": completion.choices[0].message.content}