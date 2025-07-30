# 02-GLM-4.1V-Thinking Gradio部署

THUDM也提供一个gradio界面脚本，搭建一个可以直接使用的 Web 界面，支持图片，视频，PDF，PPT等多模态输入。当然，如果glm4.1v在本地调用的及修改对应的模型路径即可。

![5b6cb0ad-cd47-451c-9f12-fda3e8300842.png](images/5b6cb0ad-cd47-451c-9f12-fda3e8300842.png)

```bash
python /root/autodl-tmp/GLM-4.1V-Thinking/inference/trans_infer_gradio.py
```

使用AutoDL的云端机器的小伙伴根据自己系统的指引连接即可

```bash
ssh -F /dev/null -CNg -L 7860:127.0.0.1:7860 [root@connect.nma1.seetacloud.com](mailto:root@connect.nma1.seetacloud.com) -p 36185
```

## 启动示例

---

![4cfaea13-f4c7-4a95-b6a3-1ab88794f204.png](images/4cfaea13-f4c7-4a95-b6a3-1ab88794f204.png)

![7146cb1a-1e79-41b8-8927-62b1ae054cff.png](images/7146cb1a-1e79-41b8-8927-62b1ae054cff.png)

![5d1abd7a-f6f1-480e-8613-6e7e4d65734b.png](images/5d1abd7a-f6f1-480e-8613-6e7e4d65734b.png)

![4d52e632-038f-487b-ae36-c59f2e3d0481.png](images/4d52e632-038f-487b-ae36-c59f2e3d0481.png)