import os
import tempfile
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import uvicorn

app = FastAPI(title="Voice Transcription API")

# Configure CORS to allow requests from the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = whisper.load_model("base")
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")

class TranscriptionResponse(BaseModel):
    success: bool
    message: str
    text: Optional[str] = None

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    if not model:
        return TranscriptionResponse(
            success=False,
            message="Whisper model not loaded correctly",
            text=None
        )
    
    try:
        # Create a temporary file to save the uploaded audio
        temp_dir = tempfile.gettempdir()
        file_id = str(uuid.uuid4())
        audio_path = os.path.join(temp_dir, f"recording_{file_id}.wav")
        
        # Save uploaded file to the temporary location
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe the audio file with whisper
        result = model.transcribe(audio_path)
        transcript = result["text"]
        
        # Clean up the temporary file
        try:
            os.remove(audio_path)
        except:
            pass
        
        return TranscriptionResponse(
            success=True,
            message="Transcription completed successfully",
            text=transcript
        )
        
    except Exception as e:
        return TranscriptionResponse(
            success=False,
            message=f"Error during transcription: {str(e)}",
            text=None
        )

@app.get("/health/")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)