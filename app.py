import os
import sys
import logging
import base64
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from typing import Optional

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler('tts_application.log')  # Output to file
    ]
)
logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self, model_name="tts_models/en/vctk/vits"):
        """
        Initialize the Text-to-Speech engine with extensive logging
        """
        try:
            logger.info("Starting TTS Model Initialization")
            
            # System Diagnostics
            logger.info(f"Python Version: {sys.version}")
            logger.info(f"Current Working Directory: {os.getcwd()}")
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            
            # Get model manager and download models
            self.model_manager = ModelManager()
            
            logger.info(f"Attempting to download model: {model_name}")
            self.model_path, self.config_path, self.model_item = self.model_manager.download_model(model_name)
            
            logger.info("Initializing Synthesizer")
            self.synthesizer = Synthesizer(
                tts_checkpoint=self.model_path,
                tts_config_path=self.config_path,
                use_cuda=torch.cuda.is_available()
            )
            
            # Male speakers list
            self.male_speakers = [
                {"index": 0, "name": "p225"},
                {"index": 1, "name": "p226"},
                {"index": 2, "name": "p227"},
                {"index": 3, "name": "p228"},
                {"index": 4, "name": "p229"},
                {"index": 5, "name": "p230"}
            ]
            
            logger.info("TTS Model Successfully Initialized")
        except Exception as e:
            logger.error(f"TTS Initialization Failed: {e}")
            logger.error(f"Detailed Error: {sys.exc_info()}")
            raise

    def text_to_speech(self, text: str, speaker_idx: int = 0, speed: float = 1.0):
        """
        Convert text to speech with comprehensive logging
        """
        try:
            logger.info(f"Generating Speech: text={text}, speaker={speaker_idx}, speed={speed}")
            
            # Validate speaker index
            if speaker_idx < 0 or speaker_idx >= len(self.male_speakers):
                logger.warning(f"Invalid speaker index {speaker_idx}. Defaulting to 0.")
                speaker_idx = 0
            
            # Generate speech
            outputs = self.synthesizer.tts(
                text=text,
                speaker_id=speaker_idx,
                speed=speed
            )
            
            # Convert to WAV
            wav = np.array(outputs) * 32767
            wav = wav.astype(np.int16)
            
            # Convert to bytes
            from io import BytesIO
            import soundfile as sf
            
            audio_buffer = BytesIO()
            sf.write(audio_buffer, wav, self.synthesizer.output_sample_rate, format='wav')
            
            logger.info("Speech Generation Successful")
            return audio_buffer.getvalue()
        
        except Exception as e:
            logger.error(f"Speech Generation Error: {e}")
            logger.error(f"Detailed Error: {sys.exc_info()}")
            raise

# Create a global TTS instance
try:
    tts_engine = TextToSpeech()
except Exception as e:
    logger.critical(f"Failed to create TTS engine: {e}")
    tts_engine = None

# FastAPI Application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TTSRequest(BaseModel):
    text: str
    speed: Optional[float] = 1.0
    speaker: Optional[int] = 0

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests
    """
    logger.info(f"Incoming {request.method} request to {request.url}")
    
    try:
        response = await call_next(request)
        logger.info(f"Response Status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request Processing Error: {e}")
        raise

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if tts_engine is None:
        logger.error("TTS Engine not initialized")
        raise HTTPException(status_code=500, detail="TTS Engine not initialized")
    
    try:
        # Generate audio
        audio_bytes = tts_engine.text_to_speech(
            text=request.text, 
            speaker_idx=request.speaker, 
            speed=request.speed
        )
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"TTS Request Processed Successfully for text: {request.text}")
        
        return {
            "audio": audio_base64,
            "sample_rate": tts_engine.synthesizer.output_sample_rate,
            "speaker": tts_engine.male_speakers[request.speaker]
        }
    except Exception as e:
        logger.error(f"TTS Request Failed: {e}")
        logger.error(f"Detailed Error: {sys.exc_info()}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to list available male speakers
@app.get("/speakers")
async def list_speakers():
    logger.info("Speakers list requested")
    return {
        "male_speakers": tts_engine.male_speakers if tts_engine else []
    }

# Health check endpoint with comprehensive diagnostics
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy" if tts_engine is not None else "not initialized",
        "cuda_available": torch.cuda.is_available(),
        "model": "VCTK VITS",
        "available_male_speakers": len(tts_engine.male_speakers) if tts_engine else 0,
        "system_info": {
            "python_version": sys.version,
            "working_directory": os.getcwd()
        }
    }

# For local and deployment testing
if __name__ == "__main__":
    import uvicorn
    
    # Use environment variable for port, default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False
    )
