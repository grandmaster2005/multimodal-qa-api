import base64
import io
import os
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageText
import pytest

from main import app

client = TestClient(app)

def create_dummy_base64_image():
    # Create a simple dummy image
    img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Sample", fill=(255,255,0))
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@pytest.fixture
def dummy_image_b64():
    return create_dummy_base64_image()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_answer_image_missing_key(dummy_image_b64, monkeypatch):
    # Ensure GEMINI_API_KEY is not set
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    payload = {
        "image_base64": dummy_image_b64,
        "question": "What does the image say?"
    }
    response = client.post("/answer-image", json=payload)
    assert response.status_code == 500
    assert "GEMINI_API_KEY" in response.json()["detail"]

def test_answer_image_invalid_base64(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    payload = {
        "image_base64": "invalid_base64_string!",
        "question": "What is this?"
    }
    response = client.post("/answer-image", json=payload)
    assert response.status_code == 400
    assert "Invalid base64 image" in response.json()["detail"]

# We won't test the actual Gemini API call without a real key/mock, 
# but we can test the schema validation.
def test_answer_image_invalid_schema(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    # Missing question
    payload = {
        "image_base64": create_dummy_base64_image()
    }
    response = client.post("/answer-image", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
