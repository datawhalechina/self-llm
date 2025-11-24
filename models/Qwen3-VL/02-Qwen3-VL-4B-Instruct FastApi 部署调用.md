# Qwen3-VL-4B-Instruct FastApi Deployment and Invocation

## Environment Preparation

Basic Environment:

```
----------------
ubuntu 22.04
python 3.12
cuda 12.8
pytorch 2.8.0
----------------
```
> This article assumes that the learner has already installed the above PyTorch (cuda) environment. If not, please install it yourself.

### GPU Configuration Instructions

This tutorial is based on the **RTX 4090** graphics card for deployment. This card has 24GB of video memory, which fully meets the running requirements of the Qwen3-VL-4B-Instruct model.

![alt text](./images/01-1.png)

### Environment Installation
First, change the `pip` source to accelerate the download and install dependency packages.

```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.20.0
pip install fastapi==0.115.4
pip install uvicorn==0.32.0
pip install transformers>=4.51.0
pip install accelerate==1.1.1
pip install torchvision==0.19.0
pip install av==13.1.0
pip install qwen-vl-utils
```

![alt text](./images/01-2.png)

## Model Download

Use the `snapshot_download` function in `modelscope` to download the model. The first parameter is the model name, and the parameter `cache_dir` is the download path of the model.

Create a new `model_download.py` file, input the following code, and run `python model_download.py` to execute the download.

```python
# model_download.py
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-VL-4B-Instruct', cache_dir='/root/autodl-fs', revision='master')
```

> Note: Please remember to modify `cache_dir` to your own model download path. It is recommended to use the `/root/autodl-fs` directory, which is a persistent storage directory, and data will not be lost after restarting the machine. The actual size of the Qwen3-VL-4B-Instruct model is about **9.2GB** (including all configuration files and weight files), and the download time depends on the network speed.

![alt text](./images/01-3.png)

## Code Preparation

### API Server Code

Create an API server file `api_server_qwen3vl_simple.py`. This file contains the complete FastAPI service implementation, supporting multimodal question and answer functions for text and images.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(8)

# Create FastAPI app
app = FastAPI(title="Qwen3-VL-4B Simple API", version="1.0.0")

# Model path
model_name_or_path = '/root/autodl-fs/Qwen/Qwen3-VL-4B-Instruct'

# Initialize model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)

# Request model
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# Response model
class ChatResponse(BaseModel):
    response: str
    model: str = "Qwen3-VL-4B-Instruct"
    usage: Dict[str, int]

@app.get("/")
async def root():
    return {"message": "Qwen3-VL-4B-Instruct API Server is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "Qwen3-VL-4B-Instruct",
        "device": str(model.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        # Process messages
        messages = request.messages
        
        # Process visual information
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Calculate token usage
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = len(generated_ids_trimmed[0])
        
        return ChatResponse(
            response=response_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
```

> **Important Note**: Modify the model path in the `model_name_or_path` variable according to the actual situation.

## Start API Service

Run the following command in the terminal to start the API service:

```shell
python api_server_qwen3vl_simple.py
```

After successful startup, you will see output similar to the following:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

![alt text](./images/01-4.png)

### Test API Service

### Test Client Code

Create a test script `test_simple_api.py` to verify the function of the image Q&A API service.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

# API Service Address
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check interface"""
    print("=== Test Health Check Interface ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed")
            print(f"Model: {result.get('model')}")
            print(f"Device: {result.get('device')}")
            print(f"GPU Memory: {result.get('gpu_memory')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check exception: {e}")
        return False

def test_text_chat():
    """Test pure text chat"""
    print("\n=== Test Pure Text Chat ===")
    
    messages = [
        {
            "role": "user",
            "content": "Hello, please introduce yourself."
        }
    ]
    
    payload = {
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Text chat test successful")
            print(f"Response: {result['response']}")
            print(f"Token Usage: {result['usage']}")
            return True
        else:
            print(f"❌ Text chat test failed: {response.status_code}")
            print(f"Error Message: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Text chat test exception: {e}")
        return False

def test_image_chat():
    """Test image chat"""
    print("\n=== Test Image Chat ===")
    
    # Use online image for testing
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url
                    },
                    {
                        "type": "text",
                        "text": "Please describe the content of this image."
                    }
                ]
            }
        ]
        
        payload = {
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image chat test successful")
            print(f"Response: {result['response']}")
            print(f"Token Usage: {result['usage']}")
            return True
        else:
            print(f"❌ Image chat test failed: {response.status_code}")
            print(f"Error Message: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Image chat test exception: {e}")
        return False

def main():
    """Main test function"""
    print("Start testing Qwen3-VL-4B-Instruct API Service")
    print("=" * 50)
    
    # Execute tests
    health_ok = test_health_check()
    text_ok = test_text_chat()
    image_ok = test_image_chat()
    
    # Summarize test results
    print("\n" + "=" * 50)
    print("Test Result Summary:")
    print(f"Health Check: {'✅ Passed' if health_ok else '❌ Failed'}")
    print(f"Text Chat: {'✅ Passed' if text_ok else '❌ Failed'}")
    print(f"Image Chat: {'✅ Passed' if image_ok else '❌ Failed'}")
    
    if health_ok and text_ok:
        print("\n🎉 API service is running normally!")
    else:
        print("\n⚠️  Some functions have problems, please check the service status")

if __name__ == "__main__":
    main()
```

> **Important Note**: This test script uses an online image link for testing, no local image file is required, making it easier to use. Test image source: `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`

The returned result after execution is as follows:

![alt text](./images/01-5.png)


## FAQ

### Q1: Model Loading Failed
**Problem**: "CUDA out of memory" error occurs
**Solution**: 
- Ensure RTX 4090 has enough video memory space
- Try using quantization configuration to reduce video memory usage
- Check if other programs are occupying video memory

### Q2: Slow Inference Speed
**Problem**: Model inference response time is too long
**Solution**:
- Reduce `max_tokens` parameter value
- Use quantized model
- Ensure CUDA and PyTorch versions are compatible


# Advanced: Video Q&A Function

## Video Q&A API Service

In addition to basic image Q&A functions, Qwen3-VL-4B-Instruct also supports video content understanding. We can create an enhanced API service to support video input.

### Create Video Q&A Service

Create a new `api_server_qwen3vl_video.py` file:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(8)

# Create FastAPI app
app = FastAPI(title="Qwen3-VL-4B Video API", version="1.0.0")

# Model path
model_name_or_path = '/root/autodl-fs/Qwen/Qwen3-VL-4B-Instruct'

# Initialize model and processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)

# Request model
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# Response model
class ChatResponse(BaseModel):
    response: str
    model: str = "Qwen3-VL-4B-Instruct"
    usage: Dict[str, int]

@app.get("/")
async def root():
    return {"message": "Qwen3-VL-4B-Instruct Video API Server is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "Qwen3-VL-4B-Instruct",
        "device": str(model.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
        "supported_formats": ["image", "video"]
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        # Process messages
        messages = request.messages
        
        # Process visual information (including images and videos)
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Calculate token usage
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = len(generated_ids_trimmed[0])
        
        return ChatResponse(
            response=response_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

# Compatible with the original /generate interface
@app.post("/generate")
async def generate_response(request: ChatRequest):
    """Compatible with original interface format"""
    result = await chat_completions(request)
    return {"response": result.response}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
```

### Start Video Q&A Service

```bash
python api_server_qwen3vl_video.py
```

![alt text](./images/01-6.png)

## Video Q&A Test

### Create Test Script

Create a new `test_video_api.py` file:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time

# API Service Address
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check interface"""
    print("=== Health Check Test ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed")
            print(f"Model: {result.get('model')}")
            print(f"Device: {result.get('device')}")
            print(f"CUDA Available: {result.get('cuda_available')}")
            print(f"GPU Memory: {result.get('gpu_memory')}")
            print(f"Supported Formats: {result.get('supported_formats')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check exception: {e}")
        return False

def test_video_conversation():
    """Test video conversation"""
    print("\n=== Video Conversation Test ===")
    try:
        # Use local video file (please ensure the video file exists)
        video_path = "./test_video.mp4"
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": 1.0,
                            "max_pixels": 360 * 420
                        },
                        {
                            "type": "text",
                            "text": "Please describe the content of this video, including main scenes and actions."
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Video conversation test successful")
            print(f"Video File: {video_path}")
            print(f"Response: {result['response']}")
            print(f"Token Usage: {result['usage']}")
            return True
        else:
            print(f"❌ Video conversation test failed: {response.status_code}")
            print(f"Error Message: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Video conversation test exception: {e}")
        return False

def main():
    """Main test function"""
    print("Start testing Qwen3-VL-4B-Instruct Video API")
    print("=" * 50)
    
    # Wait for service startup
    print("Waiting for API service to start...")
    time.sleep(2)
    
    # Execute tests
    tests = [
        test_health_check,
        test_video_conversation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        time.sleep(1)  # Test interval
    
    print("\n" + "=" * 50)
    print(f"Test Completed: {passed}/{total} Passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed, please check API service status")

if __name__ == "__main__":
    main()
```

### Run Test

```bash
python test_video_api.py
```

![alt text](./images/01-7.png)