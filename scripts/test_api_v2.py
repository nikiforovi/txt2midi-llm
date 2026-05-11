import requests
import os
import time

def test_generation():
    url = "http://localhost:8000/generate"
    payload = {
        "prompt": "fast thrash metal riff in D minor",
        "tempo": 180,
        "mode": "minor",
        "root": "D",
        "strict_scale": True,
        "chromaticity": 0.0,
        "temperature": 0.5,
        "top_p": 0.7,
        "repetition_penalty": 1.1,
        "max_len": 1024
    }
    
    print(f"--- API v2 Integration Test ---")
    print(f"Sending request to {url}...")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            output_path = "data/generated/api_test_result.mid"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Success! MIDI received and saved to: {output_path}")
            print(f"File size: {len(response.content)} bytes")
        else:
            print(f"❌ Error: Received status code {response.status_code}")
            print(f"Details: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed to connect to server: {str(e)}")
        print("Note: Make sure the server is running (python api/server.py)")

if __name__ == "__main__":
    test_generation()
