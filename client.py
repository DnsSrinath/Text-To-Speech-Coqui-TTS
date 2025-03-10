# client.py
import requests
import base64
import os

def tts_request(text, api_url, speed=1.0, speaker=None):
    """
    Send a request to the TTS API and save the returned audio
    
    Args:
        text (str): Text to convert to speech
        api_url (str): URL of the TTS API
        speed (float): Speed of speech (1.0 is normal)
        speaker (int, optional): Speaker ID for multi-speaker models
    
    Returns:
        str: Path to the saved audio file
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
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    # Parse response
    data = response.json()
    
    if 'error' in data:
        print(f"API Error: {data['error']}")
        return None
    
    # Decode the base64 audio data
    audio_data = base64.b64decode(data['audio'])
    
    # Save to file
    output_file = "tts_output.wav"
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"Audio saved to: {output_file}")
    return output_file

# Usage example
if __name__ == "__main__":
    # Replace with your deployed API URL
    API_URL = "https://your-tts-api.onrender.com"
    
    # Example: Generate speech
    text = "Hello, this is a test of the TTS API. It converts text to speech in the cloud."
    tts_request(text, API_URL)
