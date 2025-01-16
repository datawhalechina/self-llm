import requests

url = "http://localhost:8000/generate"
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "./space_woaudio.mp4"
                },
                {
                    "type": "text",
                    "text": "Describe this video."
                }
            ]
        }
    ]
}

response = requests.post(url, json=payload)
print(response.json())