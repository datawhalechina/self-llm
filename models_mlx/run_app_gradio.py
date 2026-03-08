"""
Gradio 交互式应用
功能：模型下载、模型对话（支持 MLX / Transformers 双框架）
启动: python run_app_gradio.py
"""

import os
import time
import gradio as gr

from modules.core_types import Framework
from modules.download_model import (
    get_companies, get_series, get_models,
    get_repo_id, get_local_path, model_exists, download,
    scan_local_models, get_framework_inference,
)
from modules.framework import create_backend


# ============================================================
# 📥 Tab 1: 模型下载 — 联动回调
# ============================================================

def on_source_change(source):
    """切换来源 → 更新公司列表 → 重置后续"""
    companies = get_companies(source)
    first_company = companies[0] if companies else None
    series_update, models_update, status, btn = _cascade_from_company(first_company, source)
    return (
        gr.update(choices=companies, value=first_company),
        series_update,
        models_update,
        status,
        btn,
    )


def on_company_change(company, source):
    """切换公司 → 更新系列 → 更新模型 → 检测状态"""
    return _cascade_from_company(company, source)


def _cascade_from_company(company, source):
    """从公司开始级联更新"""
    if not company:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            "", gr.update(interactive=False, value="确认下载"),
        )
    series_list = get_series(company, source)
    first_series = series_list[0] if series_list else None
    models_update, status, btn = _cascade_from_series(company, first_series, source)
    return (
        gr.update(choices=series_list, value=first_series),
        models_update,
        status,
        btn,
    )


def on_series_change(company, series, source):
    """切换系列 → 更新模型 → 检测状态"""
    return _cascade_from_series(company, series, source)


def _cascade_from_series(company, series, source):
    """从系列开始级联更新"""
    if not series:
        return (
            gr.update(choices=[], value=None),
            "", gr.update(interactive=False, value="确认下载"),
        )
    models = get_models(company, series, source)
    first_model = models[0] if models else None
    status, btn = check_model_status(first_model, source, company, series)
    return gr.update(choices=models, value=first_model), status, btn


def check_model_status(model_name, source, company=None, series=None):
    """检测模型本地是否已存在"""
    if not model_name:
        return "", gr.update(interactive=False, value="确认下载")

    repo_id = get_repo_id(model_name, source)
    local_path = get_local_path(model_name, source, company, series)

    if model_exists(model_name, source, company, series):
        return (
            f"✅ 模型已存在本地，无需下载\n\n"
            f"📋 Repo ID: {repo_id}\n"
            f"📂 本地路径: {local_path}\n\n"
            f"💡 可直接切换到「模型对话」Tab 使用该模型。"
        ), gr.update(interactive=False, value="已存在，无需下载")
    else:
        return (
            f"📋 Repo ID: {repo_id}\n"
            f"📂 将保存到: {local_path}\n\n"
            f"⚠️ 本地未检测到该模型，点击「确认下载」开始下载。"
        ), gr.update(interactive=True, value="确认下载")


def download_model(model_name, source, company, series):
    """下载模型（Gradio generator 包装）"""
    if not model_name:
        yield "⚠️ 请先选择模型"
        return

    repo_id = get_repo_id(model_name, source)
    local_path = get_local_path(model_name, source, company, series)

    if model_exists(model_name, source, company, series):
        yield f"✅ 模型已存在本地，无需重复下载\n📂 路径: {local_path}"
        return

    yield f"⏳ 开始下载 {repo_id}\n📂 保存路径: {local_path}\n\n请耐心等待..."

    try:
        _, elapsed = download(model_name, source, company, series)
        yield (
            f"✅ 下载完成！\n\n"
            f"📋 Repo ID: {repo_id}\n"
            f"📂 本地路径: {local_path}\n"
            f"⏱️ 耗时: {elapsed:.2f} 秒\n\n"
            f"💡 请切换到「模型对话」Tab，点击刷新按钮即可看到新模型。"
        )
    except Exception as e:
        yield f"❌ 下载失败: {e}"


# ============================================================
# 💬 Tab 2: 模型对话
# ============================================================

current_backend = {"instance": None, "name": None}

# ---- 本地模型多级筛选 ----

def _get_local_cache():
    """获取本地模型列表（缓存友好）"""
    return scan_local_models()


def get_local_sources():
    """获取本地已下载模型的所有来源"""
    return sorted(set(m["source"] for m in _get_local_cache()))


def get_local_companies(source):
    """获取指定来源下的公司列表"""
    return sorted(set(m["company"] for m in _get_local_cache() if m["source"] == source))


def get_local_series(source, company):
    """获取指定来源+公司下的系列列表"""
    return sorted(set(
        m["series"] for m in _get_local_cache()
        if m["source"] == source and m["company"] == company
    ))


def get_local_models(source, company, series):
    """获取指定来源+公司+系列下的模型列表（返回 (label, path) 元组）"""
    return [
        (m["model"], m["path"]) for m in _get_local_cache()
        if m["source"] == source and m["company"] == company and m["series"] == series
    ]


def _get_fw_update(source, company, series):
    """根据配置中的 FrameworkInference 生成框架 Radio 更新"""
    if source and company and series:
        fw_list = get_framework_inference(company, series, source)
    elif source:
        fw_list = [Framework.MLX] if source == "mlx" else [Framework.TRANSFORMERS]
    else:
        fw_list = list(Framework)
    choices = [(fw.value, fw.value) for fw in fw_list]
    return gr.update(choices=choices, value=fw_list[0].value if fw_list else None)


def init_chat_tab():
    """初始化/刷新对话 Tab 的所有下拉框"""
    sources = get_local_sources()
    source = sources[0] if sources else None
    companies = get_local_companies(source) if source else []
    company = companies[0] if companies else None
    series_list = get_local_series(source, company) if company else []
    series = series_list[0] if series_list else None
    models = get_local_models(source, company, series) if series else []
    model_val = models[0][1] if models else None
    return (
        gr.update(choices=sources, value=source),
        gr.update(choices=companies, value=company),
        gr.update(choices=series_list, value=series),
        gr.update(choices=models, value=model_val),
        _get_fw_update(source, company, series),
    )


def on_chat_source_change(source):
    """对话 Tab：切换来源 → 级联更新"""
    companies = get_local_companies(source)
    company = companies[0] if companies else None
    series_up, model_up, fw_up = _chat_cascade_company(source, company)
    return (
        gr.update(choices=companies, value=company),
        series_up, model_up, fw_up,
    )


def on_chat_company_change(source, company):
    """对话 Tab：切换公司 → 级联更新"""
    return _chat_cascade_company(source, company)


def _chat_cascade_company(source, company):
    if not company:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            _get_fw_update(source, None, None),
        )
    series_list = get_local_series(source, company)
    first_series = series_list[0] if series_list else None
    model_up, fw_up = _chat_cascade_series(source, company, first_series)
    return gr.update(choices=series_list, value=first_series), model_up, fw_up


def on_chat_series_change(source, company, series):
    """对话 Tab：切换系列 → 更新模型"""
    return _chat_cascade_series(source, company, series)


def _chat_cascade_series(source, company, series):
    if not series:
        return gr.update(choices=[], value=None), _get_fw_update(source, company, None)
    models = get_local_models(source, company, series)
    first_val = models[0][1] if models else None
    return gr.update(choices=models, value=first_val), _get_fw_update(source, company, series)


def load_model(model_path, framework):
    """加载模型"""
    if not model_path:
        return "⚠️ 请先选择模型"

    if not os.path.exists(model_path):
        return f"❌ 模型路径不存在: {model_path}"

    model_name = os.path.basename(model_path)

    try:
        s = time.time()
        backend = create_backend(framework)
        backend.load(model_path)
        elapsed = time.time() - s

        current_backend["instance"] = backend
        current_backend["name"] = model_name

        return (
            f"✅ 模型加载成功！\n\n"
            f"📋 模型: {model_name}\n"
            f"📂 路径: {model_path}\n"
            f"🔧 框架: {framework}\n"
            f"⏱️ 耗时: {elapsed:.2f} 秒\n\n"
            f"现在可以开始对话了 👇"
        )

    except Exception as e:
        current_backend["instance"] = None
        return f"❌ 加载失败: {e}"


def chat(message, history, temperature, top_p, max_tokens, enable_thinking):
    """处理对话"""
    backend = current_backend["instance"]
    if not backend or not backend.is_loaded:
        yield "⚠️ 请先在上方加载模型"
        return

    def extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    messages = [{"role": "system", "content": "你是一个智能助手。"}]
    for msg in history:
        messages.append({"role": msg["role"], "content": extract_text(msg["content"])})
    messages.append({"role": "user", "content": extract_text(message)})

    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if not enable_thinking:
        template_kwargs["enable_thinking"] = False

    prompt = backend.tokenizer.apply_chat_template(messages, **template_kwargs)

    yield from backend.generate(prompt, temperature, top_p, max_tokens)


# ============================================================
# 🎨 构建 UI
# ============================================================

def init_download_tab(source="mlx"):
    """页面加载/刷新时，从 JSON 重新读取并初始化所有下拉框（热加载）"""
    companies = get_companies(source)
    company = companies[0] if companies else None
    series_list = get_series(company, source) if company else []
    series = series_list[0] if series_list else None
    models = get_models(company, series, source) if series else []
    model = models[0] if models else None
    status, btn = check_model_status(model, source, company, series)
    return (
        gr.update(choices=companies, value=company),
        gr.update(choices=series_list, value=series),
        gr.update(choices=models, value=model),
        status,
        btn,
    )


def build_ui():
    with gr.Blocks(title="LLM on Mac - 本地大模型交互平台") as demo:
        gr.Markdown(
            "# LLM on Mac\n"
            "本地大模型交互平台，支持模型下载与对话（MLX / Transformers）"
        )

        with gr.Tabs():
            # ==================== Tab 1: 模型下载 ====================
            with gr.Tab("📥 模型下载"):
                gr.Markdown("### 从 HuggingFace 下载模型到本地")

                with gr.Row():
                    with gr.Column(scale=1):
                        dl_source = gr.Radio(
                            choices=["mlx", "original"],
                            value="mlx",
                            label="模型来源",
                            info="mlx: 已量化的 MLX 格式（推荐 Mac）| original: 原始 HuggingFace 模型",
                        )
                        dl_company = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="公司/组织",
                        )
                        dl_series = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="模型系列",
                        )
                        dl_model = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="选择模型",
                        )
                        with gr.Row():
                            dl_btn = gr.Button("确认下载", variant="primary", scale=2)
                            dl_refresh_btn = gr.Button("🔄 刷新", scale=1)

                    with gr.Column(scale=1):
                        dl_output = gr.Textbox(
                            label="下载状态", lines=10, interactive=False,
                        )

                # ---- 联动事件 ----

                # 刷新按钮
                dl_refresh_btn.click(
                    fn=init_download_tab,
                    inputs=[dl_source],
                    outputs=[dl_company, dl_series, dl_model, dl_output, dl_btn],
                )

                # 切换来源
                dl_source.input(
                    fn=on_source_change,
                    inputs=[dl_source],
                    outputs=[dl_company, dl_series, dl_model, dl_output, dl_btn],
                )

                # 切换公司
                dl_company.input(
                    fn=on_company_change,
                    inputs=[dl_company, dl_source],
                    outputs=[dl_series, dl_model, dl_output, dl_btn],
                )

                # 切换系列
                dl_series.input(
                    fn=on_series_change,
                    inputs=[dl_company, dl_series, dl_source],
                    outputs=[dl_model, dl_output, dl_btn],
                )

                # 切换模型 → 检测状态（需要 company, series）
                dl_model.input(
                    fn=check_model_status,
                    inputs=[dl_model, dl_source, dl_company, dl_series],
                    outputs=[dl_output, dl_btn],
                )

                # 下载（需要 company, series）
                dl_btn.click(
                    fn=download_model,
                    inputs=[dl_model, dl_source, dl_company, dl_series],
                    outputs=dl_output,
                )

            # ==================== Tab 2: 模型对话 ====================
            with gr.Tab("💬 模型对话"):
                gr.Markdown("### 加载模型")

                with gr.Row():
                    with gr.Column(scale=1):
                        chat_source = gr.Dropdown(
                            choices=[], value=None, label="模型来源",
                        )
                        chat_company = gr.Dropdown(
                            choices=[], value=None, label="公司/组织",
                        )
                        chat_series = gr.Dropdown(
                            choices=[], value=None, label="模型系列",
                        )
                        chat_model = gr.Dropdown(
                            choices=[], value=None, label="选择模型",
                        )
                    with gr.Column(scale=1):
                        chat_framework = gr.Radio(
                            choices=[(fw.value, fw.value) for fw in Framework],
                            value=Framework.MLX.value,
                            label="推理框架",
                        )
                        with gr.Row():
                            load_btn = gr.Button("加载模型", variant="primary", scale=2)
                            refresh_btn = gr.Button("🔄 刷新模型", scale=1)
                            refresh_config_btn = gr.Button("🔄 刷新配置", scale=1)

                load_status = gr.Textbox(label="加载状态", lines=4, interactive=False)

                gr.Markdown("### 对话参数")
                with gr.Row():
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Top-p")
                    max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max Tokens")
                    enable_thinking = gr.Checkbox(value=False, label="启用思考模式")

                gr.Markdown("### 开始对话")
                chatbot = gr.Chatbot(label="对话", height=400)

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题...",
                        label="输入", scale=4, show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空对话")
                    example_btns = []
                    for ex in ["请用一句话解释什么是人工智能？", "用Python写一个快速排序", "介绍MLX框架"]:
                        example_btns.append(gr.Button(ex, variant="secondary", size="sm"))

                # ---- 事件绑定 ----

                # 刷新模型列表
                refresh_btn.click(
                    fn=init_chat_tab,
                    outputs=[chat_source, chat_company, chat_series, chat_model, chat_framework],
                )

                # 刷新配置（重新读取 JSON 中的 FrameworkInference）
                refresh_config_btn.click(
                    fn=_get_fw_update,
                    inputs=[chat_source, chat_company, chat_series],
                    outputs=chat_framework,
                )

                # 级联筛选
                chat_source.input(
                    fn=on_chat_source_change,
                    inputs=[chat_source],
                    outputs=[chat_company, chat_series, chat_model, chat_framework],
                )
                chat_company.input(
                    fn=on_chat_company_change,
                    inputs=[chat_source, chat_company],
                    outputs=[chat_series, chat_model, chat_framework],
                )
                chat_series.input(
                    fn=on_chat_series_change,
                    inputs=[chat_source, chat_company, chat_series],
                    outputs=[chat_model, chat_framework],
                )

                # 加载模型
                load_btn.click(fn=load_model, inputs=[chat_model, chat_framework], outputs=load_status)

                def user_send(message, history):
                    if not message.strip():
                        return "", history
                    history = history + [{"role": "user", "content": message}]
                    return "", history

                def bot_respond(history, temperature, top_p, max_tokens, enable_thinking):
                    if not history:
                        return history
                    user_message = history[-1]["content"]
                    prev_history = history[:-1]
                    history = history + [{"role": "assistant", "content": ""}]
                    for partial in chat(user_message, prev_history, temperature, top_p, max_tokens, enable_thinking):
                        history[-1]["content"] = partial
                        yield history

                msg_input.submit(
                    fn=user_send, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot],
                ).then(
                    fn=bot_respond,
                    inputs=[chatbot, temperature, top_p, max_tokens, enable_thinking],
                    outputs=chatbot,
                )
                send_btn.click(
                    fn=user_send, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot],
                ).then(
                    fn=bot_respond,
                    inputs=[chatbot, temperature, top_p, max_tokens, enable_thinking],
                    outputs=chatbot,
                )

                clear_btn.click(fn=lambda: [], outputs=chatbot)

                for btn in example_btns:
                    btn.click(
                        fn=lambda ex: (ex, []),
                        inputs=[btn],
                        outputs=[msg_input, chatbot],
                    ).then(
                        fn=user_send, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot],
                    ).then(
                        fn=bot_respond,
                        inputs=[chatbot, temperature, top_p, max_tokens, enable_thinking],
                        outputs=chatbot,
                    )

        # 页面加载/刷新时热加载所有下拉框
        demo.load(
            fn=init_download_tab,
            inputs=[dl_source],
            outputs=[dl_company, dl_series, dl_model, dl_output, dl_btn],
        )
        demo.load(
            fn=init_chat_tab,
            outputs=[chat_source, chat_company, chat_series, chat_model, chat_framework],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
