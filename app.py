import os
import sys
import base64
import torch
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.io.wavfile import write as write_wav
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

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
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            try:
                cls._instance._initialize()
            except Exception as e:
                logger.error(f"TTS Initialization Error: {e}")
                raise
        return cls._instance

    def _initialize(self):
        """
        Initialize the Text-to-Speech engine with Coqui TTS models.
        """
        try:
            logger.info("Starting TTS Model Initialization")
            
            # System and Environment Diagnostics
            logger.info(f"Python Version: {sys.version}")
            logger.info(f"Current Working Directory: {os.getcwd()}")
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            
            # Get model manager and download models if not exists
            self.model_manager = ModelManager()
            
            # Default model
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            vocoder_name = "vocoder_models/en/ljspeech/multiband-melgan"
            
            logger.info(f"Attempting to download model: {model_name}")
            logger.info(f"Attempting to download vocoder: {vocoder_name}")
            
            # Get model paths
            self.model_path, self.config_path, self.model_item = self.model_manager.download_model(model_name)
            
            # Download vocoder
            self.vocoder_path, self.vocoder_config_path, self.vocoder_item = self.model_manager.download_model(vocoder_name)
            
            logger.info("Initializing Synthesizer")
            # Initialize synthesizer
            self.synthesizer = Synthesizer(
                tts_checkpoint=self.model_path,
                tts_config_path=self.config_path,
                vocoder_checkpoint=self.vocoder_path,
                vocoder_config=self.vocoder_config_path,
                use_cuda=torch.cuda.is_available()
            )
            
            logger.info(f"TTS Model Initialized: {model_name}")
            logger.info(f"Vocoder Initialized: {vocoder_name}")
            logger.info(f"CUDA Usage: {torch.cuda.is_available()}")
        except Exception as e:
            logger.error(f"Initialization Failed: {e}")
            raise

    def text_to_speech(self, text, speaker_idx=None, speed=1.0):
        """
        Convert text to speech and return byte array.
        """
        try:
            logger.info(f"Generating speech for text: {text}")
            
            # Generate speech
            outputs = self.synthesizer.tts(
                text=text,
                speaker_id=speaker_idx,
                speed=speed
            )
            
            # Convert float array to int16 for WAV file
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

# Create a global TTS instance during module import
try:
    tts_engine = TextToSpeech()
except Exception as e:
    logger.critical(f"Failed to create TTS engine: {e}")
    tts_engine = None

# FastAPI Application
app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    speaker: int = None

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if tts_engine is None:
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
        
        return {
            "audio": audio_base64,
            "sample_rate": tts_engine.synthesizer.output_sample_rate
        }
    except Exception as e:
        logger.error(f"TTS Request Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Diagnostic endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "tts_engine": "initialized" if tts_engine is not None else "not initialized",
        "cuda_available": torch.cuda.is_available()
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
