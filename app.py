import os
import torch
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from scipy.io.wavfile import write as write_wav
import numpy as np

class TextToSpeech:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC", vocoder_name="vocoder_models/en/ljspeech/multiband-melgan"):
        """
        Initialize the Text-to-Speech engine with Coqui TTS models.
        
        Args:
            model_name (str): Path to the TTS model
            vocoder_name (str): Path to the vocoder model
        """
        # Get model manager and download models if not exists
        self.model_manager = ModelManager()
        
        # Get model paths
        self.model_path, self.config_path, self.model_item = self.model_manager.download_model(model_name)
        self.vocoder_path, self.vocoder_config_path, self.vocoder_item = None, None, None
        
        # Download vocoder if provided
        if vocoder_name is not None:
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
    
    def text_to_speech(self, text, output_file="output.wav", speaker_idx=None, speed=1.0):
        """
        Convert text to speech and save as WAV file.
        
        Args:
            text (str): Text to convert to speech
            output_file (str): Output WAV file path
            speaker_idx (int, optional): Speaker ID for multi-speaker models
            speed (float): Speed of speech (1.0 is normal speed)
        
        Returns:
            str: Path to the generated output file
        """
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
        
        # Save to file
        write_wav(output_file, self.synthesizer.output_sample_rate, wav)
        print(f"Audio saved to: {output_file}")
        
        return output_file
    
    def list_available_models(self):
        """List all available TTS models"""
        print("Available TTS Models:")
        for model in self.model_manager.list_tts_models():
            print(f"- {model}")
        
        print("\nAvailable Vocoder Models:")
        for vocoder in self.model_manager.list_vocoder_models():
            print(f"- {vocoder}")


# Example usage
if __name__ == "__main__":
    # Initialize TTS with default model (English)
    tts = TextToSpeech()
    
    # Generate speech
    text = "Hello, this is a sample text to speech conversion using Coqui TTS in Python."
    tts.text_to_speech(text, "hello_world.wav")
    
    # Generate speech with different speed
    tts.text_to_speech("This is a slower speech sample.", "slow_speech.wav", speed=0.8)
    
    # To use a different model, you can initialize with a different model name
    # For example, a multi-speaker model:
    # tts_multi = TextToSpeech("tts_models/en/vctk/vits")
    # tts_multi.text_to_speech("This is speaker 0", "speaker0.wav", speaker_idx=0)
    # tts_multi.text_to_speech("This is speaker 1", "speaker1.wav", speaker_idx=1)
    
    # List available models
    # tts.list_available_models()
