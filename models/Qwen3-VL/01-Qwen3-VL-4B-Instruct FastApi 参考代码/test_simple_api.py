#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

# API服务地址
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("=== 测试健康检查接口 ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ 健康检查通过")
            print(f"模型: {result.get('model')}")
            print(f"设备: {result.get('device')}")
            print(f"GPU内存: {result.get('gpu_memory')}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_text_chat():
    """测试纯文本对话"""
    print("\n=== 测试纯文本对话 ===")
    
    messages = [
        {
            "role": "user",
            "content": "你好，请介绍一下你自己。"
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
            print("✅ 文本对话测试成功")
            print(f"回复: {result['response']}")
            print(f"Token使用: {result['usage']}")
            return True
        else:
            print(f"❌ 文本对话测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 文本对话测试异常: {e}")
        return False

def test_image_chat():
    """测试图像对话"""
    print("\n=== 测试图像对话 ===")
    
    # 使用在线图片进行测试
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
                        "text": "请描述这张图片的内容。"
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
            print("✅ 图像对话测试成功")
            print(f"回复: {result['response']}")
            print(f"Token使用: {result['usage']}")
            return True
        else:
            print(f"❌ 图像对话测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 图像对话测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试 Qwen3-VL-4B-Instruct API 服务")
    print("=" * 50)
    
    # 执行测试
    health_ok = test_health_check()
    text_ok = test_text_chat()
    image_ok = test_image_chat()
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"健康检查: {'✅ 通过' if health_ok else '❌ 失败'}")
    print(f"文本对话: {'✅ 通过' if text_ok else '❌ 失败'}")
    print(f"图像对话: {'✅ 通过' if image_ok else '❌ 失败'}")
    
    if health_ok and text_ok:
        print("\n🎉 API服务运行正常！")
    else:
        print("\n⚠️  部分功能存在问题，请检查服务状态")

if __name__ == "__main__":
    main()