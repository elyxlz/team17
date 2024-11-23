import requests

# Replace with your actual file path
file_path = "/Users/julienblanchon/Downloads/sdokdsopsd.wav"

# API endpoint
url = "http://0.0.0.0:8000/speech-to-text-file/"

# Open the file and make the POST request
with open(file_path, "rb") as audio_file:
    files = {"audio_file": (file_path.split("/")[-1], audio_file, "audio/wav")}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        print("Response:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error:", e)
