import os.path

import streamlit as st
import torch
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
from transformers import AutoModel, AutoTokenizer

# 模型路径 - 请确保该路径存在且包含正确的模型文件
model_path = '/root/autodl-tmp/OpenBMB/MiniCPM-o-2_6'
upload_path = "/root/autodl-tmp/upload"

# 如果上传目录不存在则创建
os.makedirs(upload_path, exist_ok=True)

# 用户和助手的聊天界面名称
U_NAME = "User"
A_NAME = "Assistant"

# 设置Streamlit页面配置
st.set_page_config(
    page_title="Self-LLM MiniCPM-V-2_6 Streamlit",
    page_icon=":robot:",
    layout="wide"
)

# 加载模型和分词器（使用缓存以提高性能）
@st.cache_resource
def load_model_and_tokenizer():
    print(f"load_model_and_tokenizer from {model_path}")
    model = (AutoModel.from_pretrained(model_path, 
                                       trust_remote_code=True, 
                                       attn_implementation='sdpa').
             to(dtype=torch.bfloat16))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


# 如果会话状态中没有模型和分词器，则进行初始化
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer()
    st.session_state.model.eval().cuda()
    print("model and tokenizer had loaded completed!")

# 初始化会话状态中的聊天历史和媒体追踪
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.uploaded_image_list = []
    st.session_state.uploaded_image_num = 0
    st.session_state.uploaded_video_list = []
    st.session_state.uploaded_video_num = 0
    st.session_state.response = ""

# 侧边栏配置

# 在侧边栏创建标题和链接
with st.sidebar:
    st.title("[开源大模型使用指南](https://github.com/datawhalechina/self-llm.git)")
    
# 创建主标题和副标题
st.title("💬 MiniCPM-V-2_6 ChatBot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

# 创建最大长度参数滑块（0-4096，默认2048）
max_length = st.sidebar.slider("max_length", 0, 4096, 2048, step=2)

# 模型生成参数设置
repetition_penalty = st.sidebar.slider("repetition_penalty", 0.0, 2.0, 1.05, step=0.01)
top_k = st.sidebar.slider("top_k", 0, 100, 100, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.7, step=0.01)

# 清除会话历史并释放内存的按钮
buttonClean = st.sidebar.button("清除会话历史", key="clean")
if buttonClean:
    # 重置所有会话状态变量
    st.session_state.chat_history = []
    st.session_state.uploaded_image_list = []
    st.session_state.uploaded_image_num = 0
    st.session_state.uploaded_video_list = []
    st.session_state.uploaded_video_num = 0
    st.session_state.response = ""

    # 如果有GPU，清除CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 刷新界面
    st.rerun()

# 使用适当的格式显示聊天历史
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            if message["image"] is not None:
                st.image(message["image"], caption='用户上传的图片', width=512, use_container_width=False)
                continue
            elif message["video"] is not None:
                st.video(message["video"], format="video/mp4", loop=False, autoplay=False, muted=True)
                continue
            elif message["content"] is not None:
                st.markdown(message["content"])
    else:
        with st.chat_message(name="model", avatar="assistant"):
            st.markdown(message["content"])

# 模式选择下拉菜单
selected_mode = st.sidebar.selectbox("选择模式", ["文本", "单图片", "多图片", "视频"])

# 定义支持的图片格式
image_type = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# 单图片模式配置
if selected_mode == "单图片":
    uploaded_image = st.sidebar.file_uploader("上传单张图片", key=1, type=image_type,
                                              accept_multiple_files=False)
    if uploaded_image is not None:
        st.image(uploaded_image, caption='用户上传的图片', width=512, use_container_width=False)
        st.session_state.chat_history.append({"role": "user", "content": None, "image": uploaded_image, "video": None})
        st.session_state.uploaded_image_list = [uploaded_image]
        st.session_state.uploaded_image_num = 1

# 多图片模式配置
if selected_mode == "多图片":
    uploaded_image_list = st.sidebar.file_uploader("上传多张图片", key=2, type=image_type,
                                                   accept_multiple_files=True)
    uploaded_image_num = len(uploaded_image_list)

    if uploaded_image_list is not None and uploaded_image_num > 0:
        for img in uploaded_image_list:
            st.image(img, caption='用户上传的图片', width=512, use_container_width=False)
            st.session_state.chat_history.append({"role": "user", "content": None, "image": img, "video": None})
        st.session_state.uploaded_image_list = uploaded_image_list
        st.session_state.uploaded_image_num = uploaded_image_num

# 定义支持的视频格式
video_type = ['.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v']

# 重要提示：要处理较大的视频文件，请使用以下命令运行：
# streamlit run ./web_demo_streamlit-minicpmv2_6.py --server.maxUploadSize 1024
# Streamlit默认的200MB上传限制可能不足以处理视频
# 请根据可用的GPU内存调整大小

# 视频模式配置
if selected_mode == "视频":
    uploaded_video = st.sidebar.file_uploader("上传单个视频文件", 
                                              key=3, 
                                              type=video_type,
                                              accept_multiple_files=False)
    if uploaded_video is not None:
        try:
            # 正确处理视频保存路径
            video_filename = os.path.basename(uploaded_video.name)
            uploaded_video_path = os.path.join(upload_path, video_filename)
            
            # 将视频文件写入磁盘
            with open(uploaded_video_path, "wb") as vf:
                vf.write(uploaded_video.getbuffer())
            
            # 显示视频并更新会话状态
            st.video(uploaded_video_path)
            st.session_state.chat_history.append({"role": "user", "content": None, "image": None, "video": uploaded_video_path})
            st.session_state.uploaded_video_list = [uploaded_video_path]
            st.session_state.uploaded_video_num = 1
            
        except Exception as e:
            st.error(f"处理视频时出错：{str(e)}")
            print(f"错误详情：{str(e)}")

# 视频处理的最大帧数 - 如果遇到CUDA内存不足，请减少此值
MAX_NUM_FRAMES = 64

def encode_video(video_path):
    """
    对视频进行编码，以固定速率采样帧并转换为图像数组。
    实现均匀采样以在内存限制下处理较长视频。
    """
    def uniform_sample(frame_indices, num_samples):
        # 计算均匀分布的采样间隔
        gap = len(frame_indices) / num_samples
        sampled_idxs = np.linspace(gap / 2, len(frame_indices) - gap / 2, num_samples, dtype=int)
        return [frame_indices[i] for i in sampled_idxs]

    # 在CPU上初始化视频读取器
    vr = VideoReader(video_path, ctx=cpu(0))

    # 以1FPS采样帧
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = list(range(0, len(vr), sample_fps))

    # 如果帧数超过最大值，进行均匀采样
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)

    # 将帧转换为PIL图像格式
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(frame.astype('uint8')) for frame in frames]

    print('帧数：', len(frames))
    return frames

# 聊天输入处理
user_text = st.chat_input("请输入您的问题")
if user_text is not None:
    if user_text.strip() == "":
        st.warning('输入消息不能为空！', icon="⚠️")
    else:
        # 显示用户消息
        with st.chat_message(U_NAME, avatar="user"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_text,
                "image": None,
                "video": None
            })
            st.markdown(f"{U_NAME}: {user_text}")

        # 使用模型处理响应
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        content_list = []  # 存储模型输入内容的列表
        imageFile = None

        with st.chat_message(A_NAME, avatar="assistant"):
            # 处理不同的输入模式
            if selected_mode == "单图片":
                print("使用单图片模式")
                if len(st.session_state.chat_history) > 1 and len(st.session_state.uploaded_image_list) >= 1:
                    uploaded_image = st.session_state.uploaded_image_list[-1]
                    if uploaded_image:
                        imageFile = Image.open(uploaded_image).convert('RGB')
                        content_list.append(imageFile)
                else:
                    print("单图片模式：未找到图片")

            elif selected_mode == "多图片":
                print("使用多图片模式")
                if len(st.session_state.chat_history) > 1 and st.session_state.uploaded_image_num >= 1:
                    for uploaded_image in st.session_state.uploaded_image_list:
                        imageFile = Image.open(uploaded_image).convert('RGB')
                        content_list.append(imageFile)
                else:
                    print("多图片模式：未找到图片")

            elif selected_mode == "视频":
                print("使用视频模式")
                if len(st.session_state.chat_history) > 1 and st.session_state.uploaded_video_num == 1:
                    uploaded_video_path = st.session_state.uploaded_video_list[-1]
                    if uploaded_video_path:
                        with st.spinner('正在编码视频，请稍候...'):
                            frames = encode_video(uploaded_video_path)
                else:
                    print("视频模式：未找到视频")

            # 配置模型生成参数
            params = {
                'sampling': True,
                'top_p': top_p,
                'top_k': top_k,
                'temperature': temperature,
                'repetition_penalty': repetition_penalty,
                "max_new_tokens": max_length,
                "stream": True
            }

            # 根据输入模式设置参数
            if st.session_state.uploaded_video_num == 1 and selected_mode == "视频":
                msgs = [{"role": "user", "content": frames + [user_text]}]
                # 视频模式特定参数
                params["max_inp_length"] = 4352  # 视频模式的最大输入长度
                params["use_image_id"] = False  # 禁用图像ID
                params["max_slice_nums"] = 1  # 如果高分辨率视频出现CUDA内存不足，请减小此值
            else:
                content_list.append(user_text)
                msgs = [{"role": "user", "content": content_list}]

            print("content_list:", content_list)  # 调试信息
            print("params:", params)  # 调试信息

            # 生成并显示模型响应
            with st.spinner('AI正在思考...'):
                response = model.chat(image=None, msgs=msgs, context=None, tokenizer=tokenizer, **params)
            st.session_state.response = st.write_stream(response)
            st.session_state.chat_history.append({
                "role": "model",
                "content": st.session_state.response,
                "image": None,
                "video": None
            })

        # 添加视觉分隔符
        st.divider()

