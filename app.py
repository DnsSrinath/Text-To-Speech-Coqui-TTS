import os
import base64
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.io.wavfile import write as write_wav
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

class TextToSpeech:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the Text-to-Speech engine with Coqui TTS models.
        """
        try:
            # Get model manager and download models if not exists
            self.model_manager = ModelManager()
            
            # Default model
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            vocoder_name = "vocoder_models/en/ljspeech/multiband-melgan"
            
            # Get model paths
            self.model_path, self.config_path, self.model_item = self.model_manager.download_model(model_name)
            
            # Download vocoder
            self.vocoder_path, self.vocoder_config_path, self.vocoder_item = self.model_manager.download_model(vocoder_name)
                
            # Initialize synthesizer
            self.synthesizer = Synthesizer(
                tts_checkpoint=self.model_path,
                tts_config_path=self.config_path,
                vocoder_checkpoint=self.vocoder_path,
                vocoder_config=self.vocoder_config_path,
                use_cuda=torch.cuda.is_available()
            )
            
            print(f"TTS Model: {model_name}")
            print(f"Vocoder: {vocoder_name}")
            print(f"Using CUDA: {torch.cuda.is_available()}")
        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def text_to_speech(self, text, speaker_idx=None, speed=1.0):
        """
        Convert text to speech and return byte array.
        
        Args:
            text (str): Text to convert to speech
            speaker_idx (int, optional): Speaker ID for multi-speaker models
            speed (float): Speed of speech (1.0 is normal speed)
        
        Returns:
            bytes: Audio data as byte array
        """
        try:
            print(f"Generating speech for: '{text}'")
            
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
            print(f"TTS Generation error: {e}")
            raise

# Create a global TTS instance during module import
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

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
