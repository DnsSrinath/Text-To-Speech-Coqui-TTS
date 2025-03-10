import os
import base64
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

class TextToSpeech:
    def __init__(self, model_name="tts_models/en/vctk/vits", 
                 vocoder_name="vocoder_models/en/ljspeech/multiband-melgan"):
        """
        Initialize the Text-to-Speech engine with Coqui TTS models.
        """
        # Get model manager and download models if not exists
        self.model_manager = ModelManager()
        
        # Get model paths
        self.model_path, self.config_path, _ = self.model_manager.download_model(model_name)
        
        # Download vocoder
        self.vocoder_path, self.vocoder_config_path, _ = self.model_manager.download_model(vocoder_name)
        
        # Initialize synthesizer
        self.synthesizer = Synthesizer(
            tts_checkpoint=self.model_path,
            tts_config_path=self.config_path,
            vocoder_checkpoint=self.vocoder_path,
            vocoder_config=self.vocoder_config_path,
            use_cuda=torch.cuda.is_available()
        )

    def text_to_speech(self, text, speaker_idx=None, speed=1.0):
        """
        Convert text to speech and return byte array.
        """
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

# Create a global TTS instance
tts_engine = TextToSpeech()

# FastAPI Application
app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    speaker: int = None

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
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
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available()
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
