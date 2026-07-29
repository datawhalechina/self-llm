## Ubuntu 24.04 / Windows 11 环境准备：ROCm 7.13 + PyTorch + vLLM（以 gfx1151 为例）

> 内容参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm) `docs/zh/01-deploy`，适配 self-llm 目录结构。完整与最新版请以 hello-rocm 为准。

**ROCm 7.13.0-preview 部署 Qwen3.5 推理框架环境准备指南。**

本节以 **Ryzen AI Max / Ryzen AI Max+（gfx1151）** 为参考，说明在 ROCm 7.13 / TheRock 体系下准备 Qwen3.5 部署环境的关键步骤。

> 官方参考：
>
> - [ROCm 7.13 安装指南（gfx1151）](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-pro-390&gfx=gfx1151)
> - [PyTorch 2.11.0 on ROCm 7.13（gfx1151）](https://rocm.docs.amd.com/en/7.13.0-preview/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.11.0&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [vLLM 0.19.1 on ROCm 7.13（gfx1151）](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html?fam=ryzen&vllm-ver=0.19.1&i=docker&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [TheRock transition guide](https://rocm.docs.amd.com/en/7.13.0-preview/about/transition-guide-TheRock.html)

---

### 一、ROCm 7.13 / TheRock 变化说明

ROCm 7.13 进入 TheRock / Core SDK 体系，核心路径和包名都发生变化：


| 项目   | 旧版 ROCm                | ROCm 7.13                                   |
| ---- | ---------------------- | ------------------------------------------- |
| 核心路径 | `/opt/rocm/`           | `/opt/rocm/core` 为核心路径                      |
| 包名前缀 | `rocm-`*、`hip*`、`roc*` | `amdrocm-*`                                 |
| 兼容性  | legacy ROCm            | Core SDK 保持 ABI / API 兼容，并通过 symlink 兼容常用路径 |


Qwen3.5 架构较新，部署时还需要注意 `vLLM`、`transformers` 是否支持 `qwen3_5` 模型类型。

---

### 二、清理已有的 ROCm / AMD 相关软件

如果系统里已经装过旧版 ROCm、旧 HIP SDK 或旧 `amdgpu-dkms`，建议先清理，避免与 ROCm 7.13 / TheRock 组件冲突：

```bash
sudo apt remove 'rocm*' 'amdrocm*' 'amdgpu-dkms*' -y
sudo apt autoremove -y
```

如果此前配置过旧的 ROCm 环境变量，也建议检查 `~/.bashrc`、`~/.zshrc`、`/etc/profile.d/` 中是否存在旧路径。

---

### 三、Ubuntu 24.04 + gfx1151 准备步骤

#### 2.1 安装 OEM kernel 6.14

```bash
sudo apt update
sudo apt install -y linux-image-6.14.0-1018-oem
sudo reboot
```

#### 2.2 安装基础依赖

```bash
sudo apt update
sudo apt install -y \
  python3.13 python3.13-venv \
  libatomic1 libquadmath0 \
  build-essential git curl wget jq pciutils
```

#### 2.3 配置 GPU 权限

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

或使用 udev 规则：

```bash
sudo tee /etc/udev/rules.d/70-amdgpu.rules <<'EOF'
KERNEL=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="render", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
sudo reboot
```

---

### 四、安装 PyTorch 2.11.0（ROCm 7.13 / gfx1151）

```bash
# 安装 uv（如已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 Python 3.13
uv python install 3.13

# 创建并激活虚拟环境
uv venv --python 3.13
source .venv/bin/activate

# 备用：使用 Python 标准库 venv
# python3.13 -m venv .venv
# source .venv/bin/activate
# python -m pip install --upgrade pip
```

```bash
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  "torch==2.11.0+rocm7.13.0" \
  "torchvision==0.26.0+rocm7.13.0" \
  "torchaudio==2.11.0+rocm7.13.0"
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

---

### 五、Windows 11 + pip 路线（ROCm 7.13）

Windows 11 上 ROCm 7.13 采用 pip / TheRock 路线。开始前需要卸载已有 HIP SDK、关闭 WDAG / SAC，并安装 AMD Software: Adrenalin Edition 26.5.1 或更新版本。

```powershell
# 安装 uv（如已安装可跳过）
irm https://astral.sh/uv/install.ps1 | iex

# 安装 Python 3.13
uv python install 3.13

# 创建并激活虚拟环境
uv venv --python 3.13
.venv\Scripts\activate

# 备用：使用 Python 标准库 venv
# py -3.13 -m venv .venv
# .venv\Scripts\activate

uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ `
  "torch==2.11.0+rocm7.13.0" `
  "torchvision==0.26.0+rocm7.13.0" `
  "torchaudio==2.11.0+rocm7.13.0"

python -c "import torch; print(torch.cuda.is_available())"
```

---

### 六、vLLM 环境验证（Docker 方式）

ROCm 7.13 官方 vLLM Docker 镜像以 gfx1151 为例：

```bash
docker pull rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

> 注意：vLLM 0.19.1 Docker 镜像内置 PyTorch 2.10.0；PyTorch 2.11.0 是 pip 安装路线。

```bash
docker run -it --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --network=host \
  --ipc=host \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v ~/models:/app/models \
  -e HF_HOME="/app/models" \
  rocm/vllm:rocm7.13.0_gfx1151_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1 \
  bash
```

容器内验证：

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'HIP:', torch.cuda.is_available())"
```

---

### 七、Qwen3.5 部署前检查

如果出现模型加载失败，优先检查：

1. `vLLM` 是否为 ROCm 7.13 对应版本或官方镜像；
2. `transformers` 是否满足 Qwen3.5 所需版本；
3. `--max-model-len` 是否过大导致显存不足；
4. API 请求中是否按需要设置 `enable_thinking`。

---

### 八、后续部署教程

- [LM Studio 部署教程](./2-lm-studio-rocm7-deploy.md)
- [Ollama 部署教程](./4-ollama-rocm7-deploy.md)
- [llama.cpp 部署教程](./5-llamacpp-rocm7-deploy.md)
- [vLLM 部署教程](./3-vllm-rocm7-deploy.md)

---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。

