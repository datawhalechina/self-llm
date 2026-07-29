## Ubuntu 24.04 / Windows 11 环境准备：ROCm 7.13 + PyTorch + vLLM（以 gfx1151 为例）

> 内容参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm) `docs/zh/01-deploy`，适配 self-llm 目录结构。完整与最新版请以 hello-rocm 为准。

**ROCm 7.13.0-preview 部署推理框架环境准备指南。**

本节以 **Ryzen AI Max / Ryzen AI Max+（gfx1151）** 为参考，说明在 ROCm 7.13 / TheRock 体系下准备 Gemma 4 部署环境的关键步骤。

> 官方参考：
>
> - [ROCm 7.13 安装指南（gfx1151）](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html?fam=ryzen&w=compute&os=windows&windows-ver=11&i=pip&gpu=max-pro-390&gfx=gfx1151)
> - [PyTorch 2.11.0 on ROCm 7.13（gfx1151）](https://rocm.docs.amd.com/en/7.13.0-preview/frameworks/pytorch/install.html?fam=ryzen&os=windows&pytorch-ver=2.11.0&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [vLLM 0.19.1 on ROCm 7.13（gfx1151）](https://rocm.docs.amd.com/en/7.13.0-preview/ai-inference/vllm.html?fam=ryzen&vllm-ver=0.19.1&i=docker&w=compute&gpu=max-pro-390&gfx=gfx1151)
> - [TheRock transition guide](https://rocm.docs.amd.com/en/7.13.0-preview/about/transition-guide-TheRock.html)

---

### 一、ROCm 7.13 / TheRock 变化说明

ROCm 7.13 进入 TheRock / Core SDK 体系，和旧版 ROCm 有几处重要差异：


| 项目   | 旧版 ROCm                | ROCm 7.13                                   |
| ---- | ---------------------- | ------------------------------------------- |
| 核心路径 | `/opt/rocm/`           | `/opt/rocm/core` 为核心路径                      |
| 包名前缀 | `rocm-`*、`hip*`、`roc*` | `amdrocm-*`                                 |
| 兼容性  | legacy ROCm            | Core SDK 保持 ABI / API 兼容，并通过 symlink 兼容常用路径 |
| 工具变化 | ROCm SMI 常见            | AMD SMI 逐步替代 ROCm SMI                       |


如果使用包管理器安装，ROCm 会配置 `update-alternatives` 和常用兼容 symlink。若使用 tarball 或自定义安装目录，需要特别关注 `PATH`、`LD_LIBRARY_PATH`、`ROCM_PATH` 是否指向 `/opt/rocm/core`。

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

gfx1151 在 Ubuntu 24.04 上需要 OEM kernel 6.14 才能正确驱动 iGPU：

```bash
sudo apt update
sudo apt install -y linux-image-6.14.0-1018-oem
sudo reboot
```

重启后确认内核：

```bash
uname -r
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

任选一种方式。

方式 A：加入用户组。

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

方式 B：添加 udev 规则。

```bash
sudo tee /etc/udev/rules.d/70-amdgpu.rules <<'EOF'
KERNEL=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", GROUP="render", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
sudo reboot
```

#### 2.4 验证 GPU 设备

```bash
ls -l /dev/kfd /dev/dri
```

如果 `/dev/kfd` 不存在，优先检查 kernel、驱动和用户组权限。

---

### 四、安装 PyTorch 2.11.0（ROCm 7.13 / gfx1151）

本项目推荐使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境和依赖，替代传统 `pip + venv` 流程。

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

安装 ROCm 7.13 对应的 PyTorch 2.11.0：

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

预期 `torch.cuda.is_available()` 输出 `True`。

---

### 五、Windows 11 + pip 路线（ROCm 7.13）

Windows 11 上 ROCm 7.13 采用 pip / TheRock 路线。开始前需要：

1. 卸载已有 HIP SDK；
2. 关闭 WDAG（Microsoft Defender Application Guard）；
3. 关闭 SAC（Smart App Control）；
4. 安装 AMD Software: Adrenalin Edition 26.5.1 或更新版本；
5. 安装 Python 3.11 / 3.12 / 3.13 / 3.14。

以 Python 3.13 为例：

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

> 注意：vLLM 0.19.1 Docker 镜像内置的是 PyTorch 2.10.0，不是 PyTorch 2.11.0。PyTorch 2.11.0 对应的是上面的 pip 安装路线，二者不要混写。

启动容器：

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

### 七、后续部署教程

- [LM Studio 部署教程](./3-lm-studio-rocm7-deploy.md)
- [Ollama 部署教程](./5-ollama-rocm7-deploy.md)
- [llama.cpp 部署教程](./6-llamacpp-rocm7-deploy.md)
- [vLLM 部署教程](./4-vllm-rocm7-deploy.md)

---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。

