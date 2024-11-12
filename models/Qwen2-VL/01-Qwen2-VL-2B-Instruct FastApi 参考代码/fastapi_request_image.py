import requests

url = "http://localhost:8000/generate"
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ]
        }
    ]
}
response = requests.post(url, json=payload)
print(response.json())