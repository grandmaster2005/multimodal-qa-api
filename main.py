import os
import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image

app = FastAPI(title="Multimodal QA API")

# Enable CORS for Cloudflare Worker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QAQuery(BaseModel):
    image_base64: str
    question: str

class QAResponse(BaseModel):
    answer: str

# Configure Gemini
# Expects GEMINI_API_KEY environment variable
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

SYSTEM_PROMPT = (
    "You are an expert data extraction assistant. You will be provided with an image and a question. "
    "Your task is to answer the question based *only* on the provided image. "
    "Rules: "
    "1. The answer MUST be a string. "
    "2. For numeric answers, return ONLY the number as a string (e.g. '4089.35'). Do not include currency symbols or units. "
    "3. Do not include extra text, conversational fillers, or explanations. Just the raw answer value."
)

@app.post("/answer-image", response_model=QAResponse)
async def answer_image(query: QAQuery):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(query.image_base64)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    try:
        # Use Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # We pass the system prompt and rules as text along with the image and question
        full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query.question}"
        
        response = model.generate_content([image, full_prompt])
        
        # Clean up response by stripping whitespace
        answer = response.text.strip().strip('"').strip("'")
        
        return QAResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}
