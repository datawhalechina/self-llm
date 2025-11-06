#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time

# API服务地址
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("=== 健康检查测试 ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ 健康检查通过")
            print(f"模型: {result.get('model')}")
            print(f"设备: {result.get('device')}")
            print(f"CUDA可用: {result.get('cuda_available')}")
            print(f"GPU内存: {result.get('gpu_memory')}")
            print(f"支持格式: {result.get('supported_formats')}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_video_conversation():
    """测试视频对话"""
    print("\n=== 视频对话测试 ===")
    try:
        # 使用本地视频文件（请确保视频文件存在）
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
                            "text": "请描述这个视频的内容，包括主要场景和动作。"
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
            print("✅ 视频对话测试成功")
            print(f"视频文件: {video_path}")
            print(f"回复: {result['response']}")
            print(f"Token使用: {result['usage']}")
            return True
        else:
            print(f"❌ 视频对话测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 视频对话测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试 Qwen3-VL-4B-Instruct Video API")
    print("=" * 50)
    
    # 等待服务启动
    print("等待API服务启动...")
    time.sleep(2)
    
    # 执行测试
    tests = [
        test_health_check,
        test_video_conversation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        time.sleep(1)  # 测试间隔
    
    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查API服务状态")

if __name__ == "__main__":
    main()