# client.py
import requests
import base64
import os

def tts_request(text, api_url, speed=1.0, speaker=None):
    """
    Send a request to the TTS API and return an audio file
    
    Args:
        text (str): Text to convert to speech
        api_url (str): URL of the TTS API
        speed (float): Speed of speech (1.0 is normal)
        speaker (int, optional): Speaker ID for multi-speaker models
    
    Returns:
        str: Path to the generated audio file
    """
    # Prepare request data
    payload = {
        'text': text,
        'speed': speed
    }
    
    if speaker is not None:
        payload['speaker'] = speaker
    
    # Send request to API
    response = requests.post(f"{api_url}/tts", json=payload)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    # Parse response
    data = response.json()
    
    # Decode the base64 audio data
    audio_data = base64.b64decode(data['audio'])
    
    # Save to file
    output_file = "output.wav"
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    return output_file

# Usage example
if __name__ == "__main__":
    # Deployed API URL
    API_URL = "https://text-to-speech-coqui-tts.onrender.com"
    
    # Example text
    text = "Hello, this is a test of the Text to Speech API."
    
    # Generate speech and get the audio file
    audio_file = tts_request(text, API_URL)
    print(f"Audio generated: {audio_file}")
