# Hunyuan3D-2系列Gradio App

官方也提供了在自己的计算机上托管 Gradio 应用程序的方式代码：

标准版本

```bash
# Hunyuan3D-2mini
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mini --subfolder hunyuan3d-dit-v2-mini-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
# Hunyuan3D-2mv
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
# Hunyuan3D-2
python3 gradio_app.py --model_path tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
```

Turbo版本

```bash
# Hunyuan3D-2mini
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mini --subfolder hunyuan3d-dit-v2-mini-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
# Hunyuan3D-2mv
python3 gradio_app.py --model_path tencent/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
# Hunyuan3D-2
python3 gradio_app.py --model_path tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0-turbo --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode --enable_flashvdm
```

需要注意的是，如果是采用指定数据盘存储模型文件的方式进行加载，以下位置需要修改为需要修改为对应的本地路径

1. ’--model_path‘ 以及 ’--texgen_model_path‘ 
2. Hunyuan3D-2/gradio.py: 648-650 
    
    ```python
    if __name__ == '__main__':
        import argparse
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default='weights/Hunyuan3D-2mini') # 修改为自己的模型文件存储路径
        parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo')
        parser.add_argument("--texgen_model_path", type=str, default='weights/Hunyuan3D-2') # 修改为自己的模型文件存储路径
        parser.add_argument('--port', type=int, default=8080)
        parser.add_argument('--host', type=str, default='0.0.0.0')
    
    ```
    
3. Hunyuan3D-2/hy3dgen/shapegen/pipeline.py:264-266 
    
    ```python
    turbo_vae_mapping = {
          'Hunyuan3D-2': ('weights/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'), # 修改为自己的模型文件存储路径
          'Hunyuan3D-2mv': ('weights/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'), # 修改为自己的模型文件存储路径
          'Hunyuan3D-2mini': ('weights/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'), # 修改为自己的模型文件存储路径
         }
    ```
    

## 启动示例

---

![image.png](image.png)

![image.png](image%201.png)

![image.png](image%202.png)