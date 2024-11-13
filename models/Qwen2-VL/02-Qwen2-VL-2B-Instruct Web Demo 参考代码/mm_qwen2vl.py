import threading
from pathlib import Path
from argparse import ArgumentParser
from threading import Thread
from typing import Dict, List, Tuple

import gradio as gr
from qwen_vl_utils import process_vision_info
import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TextIteratorStreamer,
    GenerationConfig,
)
from transformers.utils import is_flash_attn_2_available
from accelerate import init_empty_weights
from accelerate.utils import calculate_maximum_sizes, convert_bytes
from accelerate.commands.estimate import create_ascii_table

# copy from qwen_vl_utils.process_vision_info
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28

# default
DEFAULT_CKPT_PATH = "./Qwen2-VL-2B-Instruct"
VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".mpeg",
]
IMAGE_EXTENSIONS = [".png", ".jpg"]

print(f"单张图片最大/最小token长度限制:{MAX_PIXELS//28*28}/{MIN_PIXELS//28*28}")
print(
    f"单个视频最大/最小/总token长度限制:{VIDEO_MAX_PIXELS//28*28}/{VIDEO_MIN_PIXELS//28*28}/{VIDEO_TOTAL_PIXELS//28*28}"
)


def cal_model_size(args):
    # modify from https://github.com/huggingface/accelerate/blob/c0552c9012a9bae7f125e1df89cf9ee0b0d250fd/src/accelerate/commands/estimate.py#L285
    model_name = Path(args.model_path).name
    model_path = Path(args.model_path).as_posix()
    use_fa2 = (
        "flash_attention_2"
        if args.flash_attn2 and is_flash_attn_2_available()
        else None
    )
    with init_empty_weights():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", attn_implementation=use_fa2
        )
    total_size, largest_layer = calculate_maximum_sizes(model)
    data = []

    for dtype in ["float32", "float16", "int8", "int4"]:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        # dtype_training_size = estimate_training_usage(dtype_total_size, dtype)    # buxuyao
        if dtype == "float16":
            dtype_total_size /= 2
            dtype_largest_layer /= 2
        elif dtype == "int8":
            dtype_total_size /= 4
            dtype_largest_layer /= 4
        elif dtype == "int4":
            dtype_total_size /= 8
            dtype_largest_layer /= 8
        row = [dtype, dtype_largest_layer, dtype_total_size]
        for i, item in enumerate(row):
            if isinstance(item, (int, float)):
                row[i] = convert_bytes(item)
            elif isinstance(item, dict):
                training_usage = max(item.values())
                row[i] = (
                    convert_bytes(training_usage) if training_usage != -1 else "N/A"
                )
        data.append(row)

    headers = ["dtype", "Largest Layer", "Total Size"]
    title = f"Memory Usage for loading `{model_name}`"
    table = create_ascii_table(headers, data, title)
    print(table)


def _get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--model-path",
        default=DEFAULT_CKPT_PATH,
        help="模型路径, 默认为%(default)r。",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="仅CPU模式运行。(不启用则默认平均分到所有显卡上)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="加载特定类型的模型。(不启用则默认`auto`, 从config获取。其他类型请自己修改。)",
    )
    parser.add_argument(
        "--cal-size",
        action="store_true",
        help="仅输出模型显存占用。(默认输出float32、float16、int8、int4的占用)",
    )
    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        default=False,
        help="使用 `flash_attention_2` 推理。(不启用则根据环境使用eager或sdpa)",
    )
    parser.add_argument(
        "--port", type=int, default=12345, help="Demo服务器端口, 默认为`12345`。"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="DDemo服务器地址, 默认为`127.0.0.1`。"
    )

    args = parser.parse_args()
    return args


class LazyModelLoader:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.proc = None
        self.lock = threading.Lock()

    def _load_model(self):
        """加载模型和tokenizer"""
        with self.lock:
            if self.model is None:  # 确保模型只加载一次
                print(f"Loading model: {self.args.model_path}")
                try:
                    model, proc = self._load_model_processor()
                except Exception:
                    self.lock.release()
                    import traceback

                    traceback.print_exc()
                    exit(-1)
                self.model = model
                self.proc = proc
                print(f"Model {self.args.model_path} loaded")
                print(f"{model.device=}")

    def _load_model_processor(self):
        args = self.args
        device_map = "cpu" if args.cpu else "auto"
        use_fa2 = (
            "flash_attention_2"
            if args.flash_attn2 and is_flash_attn_2_available()
            else None
        )
        dtype = (
            {
                "fp16": torch.float16,
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "int4": "auto",
                "int8": "auto",  # 自己改吧
            }[args.dtype]
            if args.dtype != "auto"
            else args.dtype
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            # support eager/flash_attention_2/sdpa
            attn_implementation=use_fa2,
            # auto: distribute to every GPU.
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        return model, processor

    def get_model(self):
        """获取加载的模型，若尚未加载则触发加载"""
        if self.model is None:
            threading.Thread(target=self._load_model).start()
        return self.model

    def get_processor(self):
        """获取加载的tokenizer"""
        if self.proc is None:
            threading.Thread(target=self._load_model).start()
        return self.proc


def _transform_messages(
    messages: List[List[str | Tuple[str, ...]]],
    video_extensions=VIDEO_EXTENSIONS,
    image_extensions=IMAGE_EXTENSIONS,
    user_tag="user",
    assistant_tag="assistant",
):
    transformed_messages = [{"role": user_tag, "content": []}]
    for message in messages:
        q = message[0]
        if isinstance(q, tuple):
            for it in q:
                if Path(it).suffix in video_extensions:
                    new_item = {
                        "type": "video",
                        "video": it,
                        "min_pixels": VIDEO_MIN_PIXELS,
                        "max_pixels": VIDEO_MAX_PIXELS,
                        "total_pixels": VIDEO_TOTAL_PIXELS,
                    }
                elif Path(it).suffix in image_extensions:
                    new_item = {
                        "type": "image",
                        "image": it,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    }
                transformed_messages[-1]["content"].append(new_item)
        elif isinstance(q, str):
            if transformed_messages[-1]["content"]:
                new_item = {"type": "text", "text": it}
                transformed_messages[-1]["content"].append(new_item)
            else:
                transformed_messages[-1]["content"] = q

        if message[1]:
            transformed_messages.extend(
                [
                    {"role": assistant_tag, "content": message[1]},
                    {"role": user_tag, "content": []},
                ]
            )
    return transformed_messages


def _gc():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_like_dislike(x: gr.LikeData):
    print(f"{x.index=} {x.value=}{x.liked=}")


def add_message(history: List, message: Dict):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def _pred(
    messages,
    temperature,
    topk,
    topp,
    processor: Qwen2VLProcessor,
    model: Qwen2VLForConditionalGeneration,
):
    messages = _transform_messages(messages)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )
    _gen_kwargs = (
        dict(temperature=temperature, top_p=topp, top_k=topk)
        if temperature
        else dict(do_sample=False)
    )
    gen_config = GenerationConfig(max_new_tokens=512, **_gen_kwargs)

    thread = Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            generation_config=gen_config,
            streamer=streamer,
        ),
    )
    thread.start()

    return streamer


def bot(history: List, temperature, topk, topp):
    _gc()
    model, proc = loader.get_model(), loader.get_processor()
    stream = _pred(history, temperature, topk, topp, processor=proc, model=model)
    # stream = "Test"
    history[-1][1] = ""
    for it in stream:
        history[-1][1] += it
        yield history


def web_demo(args):
    with gr.Blocks(fill_height=True) as demo:
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                label="Qwen2VL demo",
                elem_id="chatbot",
                bubble_full_width=False,
                scale=1,
                type="tuples",
            )
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Enter message or upload file...",
                show_label=False,
            )
        with gr.Column(scale=1):
            with gr.Accordion("Gen Config", open=False):
                temperature = gr.Slider(0.0, 1.0, step=0.01, label="Temperature")
                topk = gr.Slider(-1, 1000, step=2, label="Top K")  # need?
                topp = gr.Slider(0.0, 1.0, step=0.01, label="Top P")  # need?
        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(
            bot, [chatbot, temperature, topk, topp], chatbot, api_name="bot_response"
        )
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot.like(print_like_dislike, None, None)
    demo.launch(max_threads=2, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    args = _get_args()
    print(
        "flash-attn 已安装。"
        if args.flash_attn2 and is_flash_attn_2_available()
        else "flash-attn 未安装。"
    )
    cal_model_size(args)
    if args.cal_size is False:
        loader = LazyModelLoader(args)
        loader.get_model()
        web_demo(args)
