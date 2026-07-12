import os
import base64
import io
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
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

SYSTEM_PROMPT = (
    "You are an expert data extraction assistant. You will be provided with an image and a question. "
    "Your task is to answer the question based *only* on the provided image. "
    "Rules: "
    "1. The answer MUST be a string. "
    "2. For numeric answers, return ONLY the number as a string (e.g. '4089.35'). Do not include currency symbols or units. "
    "3. Do not include extra text, conversational fillers, or explanations. Just the raw answer value."
)

@app.post("/answer-image")
async def answer_image(query: QAQuery):
    try:
        # 1. Accept both raw base64 and data:image/png;base64,... inputs
        b64_string = query.image_base64
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
            
        try:
            image_data = base64.b64decode(b64_string)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            # We must still return 500 for server misconfiguration
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")

        # 2. Use the latest supported Gemini SDK/API
        client = genai.Client(api_key=api_key)
        full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {query.question}"

        # 3. Fix generate_content() usage (passing prompt then image)
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[full_prompt, image]
            )
        except Exception as e:
            traceback.print_exc()
            # 6. Never return HTTP 500 for expected Gemini failures; return a valid JSON error message
            raise HTTPException(status_code=400, detail=f"Gemini API error: {str(e)}")

        # 4. Handle cases where response.text is empty and extract text from candidates instead
        answer = ""
        try:
            if response.text:
                answer = response.text
        except ValueError:
            pass
            
        if not answer and response.candidates:
            # Fallback to extract from candidates
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                answer = candidate.content.parts[0].text
                
        if not answer:
            answer = "Error: Could not extract answer."

        # Clean up the answer
        answer = answer.strip().strip('"').strip("'")
        
        return {"answer": answer}
        
    except HTTPException:
        raise
    except Exception as e:
        # 7. Print full tracebacks to logs
        traceback.print_exc()
        # 5. Return HTTP 400 only for invalid input/unexpected errors instead of 500
        raise HTTPException(status_code=400, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}
