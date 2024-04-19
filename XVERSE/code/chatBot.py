import argparse
import torch
import gradio as gr
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig

tokenizer, model = None, None

def init_model(args):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, truncation_side="left", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, device_map='auto')
    model.generation_config = GenerationConfig.from_pretrained(args.model_path)
    model = model.eval()

def chat(message, history, request: gr.Request):
    global tokenizer, model
    history = history or []
    history.append({"role": "user", "content": message})

    # init
    history.append({"role": "assistant", "content": ""})
    utter_history = []
    for i in range(0, len(history), 2):
        utter_history.append([history[i]["content"], history[i+1]["content"]])

    # chat with stream
    for next_text in model.chat(tokenizer, history[:-1], stream=True):
        utter_history[-1][1] += next_text
        history[-1]["content"] += next_text
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        yield utter_history, history

    # log
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{current_time} request_ip:{request.client.host}\nquery: {message}\nhistory: {json.dumps(history, ensure_ascii=False)}\nanswer: {json.dumps(utter_history[-1][1], ensure_ascii=False)}')

# 增加配置，添加模型地址
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6006,
                       help="server port")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/xverse/XVERSE-7B-Chat",
                        help="model path")
    parser.add_argument("--tokenizer_path", type=str, default="/root/autodl-tmp/xverse/XVERSE-7B-Chat",
                        help="Path to the tokenizer.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # 初始化模型
    init_model(args)

    # 构建demo应用
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
                        # <center>💬 XVERSE-7B-Chat</center>
                        ## <center>🚀 A Gradio chatbot powered by Self-LLM</center>
                        ### <center>✨ 感兴趣的小伙伴可以去看我们的开源项目哦——[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)，教你一杯奶茶跑通所有主流大模型😀。</center>
                    """)
        chatbot = gr.Chatbot(label="Chat history", height=500)
        state = gr.State([])

        with gr.Row():
            text_box = gr.Textbox(label="Message", show_label=False, placeholder="请输入你的消息并回车")

        with gr.Row():
            submit_btn = gr.Button(value="Send", variant="secondary")
            reset_btn = gr.Button(value="Reset")

        text_box.submit(fn=chat,
                        inputs=[text_box, state],
                        outputs=[chatbot, state],
                        api_name="chat")
        submit_btn.click(fn=chat,
                         inputs=[text_box, state],
                         outputs=[chatbot, state])

        # 用于清空text_box
        def clear_textbox():
            return gr.update(value="")
        text_box.submit(fn=clear_textbox, inputs=None, outputs=[text_box])
        submit_btn.click(fn=clear_textbox, inputs=None, outputs=[text_box])

        # 用于清空页面和重置state
        def reset():
            return None, []
        reset_btn.click(fn=reset, inputs=None, outputs=[chatbot, state])

    demo.launch(server_name="0.0.0.0", server_port=args.port)