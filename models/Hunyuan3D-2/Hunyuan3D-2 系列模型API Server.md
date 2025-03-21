# Hunyuan3D-2系列 API Server

Hunyuan3D-2支持在本地启动一个 API 服务器，该服务器可以将图像/文本发布到 3D、纹理化现有网格等的 Web 请求

```bash
python3 api_server.py --host 0.0.0.0 --port 8080
```

同样的，如果是采用指定数据盘存储模型文件的方式进行加载，以下位置需要修改为需要修改为对应的本地路径：

1. Hunyuan3D-2/api_server.py: class ModelWorker: 148-150
    
    ```python
        def __init__(self,
                     model_path='weights/Hunyuan3D-2mini',# 修改为自己的模型文件存储路径
                     tex_model_path='weights/Hunyuan3D-2',# 修改为自己的模型文件存储路径
                     subfolder='hunyuan3d-dit-v2-mini-turbo',
                     device='cuda',
                     enable_tex=False):
    ```
    
2. Hunyuan3D-2/api_server.py: if **name** == "**main**": 304-305
    
    ```python
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=str, default="8081")
        parser.add_argument("--model_path", type=str, default='weights/Hunyuan3D-2mini')# 修改为自己的模型文件存储路径
        parser.add_argument("--tex_model_path", type=str, default='weights/Hunyuan3D-2')# 修改为自己的模型文件存储路径
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--limit-model-concurrency", type=int, default=5)
        parser.add_argument('--enable_tex', action='store_true')
        args = parser.parse_args()
        logger.info(f"args: {args}")
    ```
    

![image.png](image.png)

向服务器发送将图像转换为无纹理的3D的请求

```bash
# 生成 Base64 字符串
img_b64_str=$(base64 -w 0 input.jpg) # 修改为自己图像输入路径

# 生成 data.json
cat <<EOF > data.json
{
  "image": "$img_b64_str"
}
EOF

# 发送请求并调试
curl -v -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     --data-binary @data.json \
     -o test2.glb # 输出glb文件命名

# 清理临时文件
rm data.json
```

可以在项目路径下得到输出的glb文件

![image.png](image%201.png)