import os
import sys
import base64
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tts_app.log')
    ]
)
logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self, 
                 model_name="tts_models/en/vctk/vits", 
                 max_download_attempts=3):
        """
        Initialize TTS with robust error handling and multiple download attempts
        """
        self.model = None
        self.synthesizer = None
        
        # Log system information
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Current Working Directory: {os.getcwd()}")
        logger.info(f"Available Environment Variables: {list(os.environ.keys())}")
        
        # Attempt to download and initialize model
        for attempt in range(max_download_attempts):
            try:
                logger.info(f"Model Initialization Attempt {attempt + 1}")
                self._initialize_model(model_name)
                break
            except Exception as e:
                logger.error(f"Initialization Error (Attempt {attempt + 1}): {e}")
                if attempt == max_download_attempts - 1:
                    raise RuntimeError(f"Failed to initialize TTS model after {max_download_attempts} attempts")

    def _initialize_model(self, model_name):
        """
        Internal method to initialize the TTS model with comprehensive logging
        """
        from TTS.utils.manage import ModelManager
        from TTS.utils.synthesizer import Synthesizer

        # Get model manager and download models
        model_manager = ModelManager()
        
        logger.info(f"Downloading TTS Model: {model_name}")
        model_path, config_path, _ = model_manager.download_model(model_name)
        
        logger.info("Initializing Synthesizer")
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=torch.cuda.is_available()
        )
        
        logger.info("Model Initialization Successful")

    def text_to_speech(self, text: str, speed: float = 1.0):
        """
        Convert text to speech with error handling
        """
        if not self.synthesizer:
            raise RuntimeError("TTS model not initialized")
        
        try:
            # Generate speech
            outputs = self.synthesizer.tts(
                text=text,
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
            
            return audio_buffer.getvalue()
        
        except Exception as e:
            logger.error(f"Speech Generation Error: {e}")
            raise

# Global TTS instance
try:
    tts_engine = TextToSpeech()
except Exception as e:
    logger.critical(f"Failed to create global TTS engine: {e}")
    tts_engine = None

# FastAPI Application
app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    speed: Optional[float] = 1.0

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if tts_engine is None:
        raise HTTPException(status_code=500, detail="TTS Engine not initialized")
    
    try:
        # Generate audio
        audio_bytes = tts_engine.text_to_speech(
            text=request.text, 
            speed=request.speed
        )
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "sample_rate": tts_engine.synthesizer.output_sample_rate
        }
    except Exception as e:
        logger.error(f"TTS Request Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy" if tts_engine is not None else "not initialized",
        "cuda_available": torch.cuda.is_available(),
        "model": "VCTK VITS",
        "environment": {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "port": os.environ.get('PORT', 'Not Set')
        }
    }

# Local testing
if __name__ == "__main__":
    import uvicorn
    
    # Use environment variable for port, with a default
    port = int(os.environ.get("PORT", 10000))
    
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False
    )
