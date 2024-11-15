import threading
from pathlib import Path
from argparse import ArgumentParser, Namespace
from threading import Thread
from typing import Any, Dict, Generator, List, Tuple

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
from transformers.modeling_utils import get_first_parameter_dtype
from accelerate import init_empty_weights
from accelerate.utils import calculate_maximum_sizes, convert_bytes
from accelerate.commands.estimate import create_ascii_table

# copy from qwen_vl_utils.process_vision_info
MIN_PIXELS = 4 * 28 * 28  # 一张图最小占4个token
MAX_PIXELS = 16384 * 28 * 28  # 一张图最大占16384个token
VIDEO_MIN_PIXELS = 128 * 28 * 28  # 一个视频里一帧最小占128个token
VIDEO_MAX_PIXELS = 768 * 28 * 28  # 一个视频里一帧最大占768个token
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28  # 一个视频里所有帧总共占最多24576个token

# default
DEFAULT_CKPT_PATH = "path/to/Qwen2-VL-2B-Instruct"
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
# end default
print("*" * 60)
print("*Qwen2-vl 图片视频模态token限制如下:")
print(f"*单张图片最大/最小token长度限制:{MAX_PIXELS//(28*28)}/{MIN_PIXELS//(28*28)}")
print(
    f"*单个视频最大/最小/总token长度限制:{VIDEO_MAX_PIXELS//(28*28)}/{VIDEO_MIN_PIXELS//(28*28)}/{VIDEO_TOTAL_PIXELS//(28*28)}"
)
print("*" * 60, end="\n\n")


# modify from https://github.com/huggingface/accelerate/blob/c0552c9012a9bae7f125e1df89cf9ee0b0d250fd/src/accelerate/commands/estimate.py#L285
def cal_model_size(args):
    """计算模型在各种数据类型下的存储占用
    主要计算方法是
    借助calculate_maximum_sizes函数计算所有参数数量在特定下的存储->float32,float16,int8,int4分别进行进一步乘除即可.
    convert_bytes: 将计算结果转为不超过1024的TB/GB/MB/KB等单位下的结果表示.
    """
    model_name = Path(args.model_path).name
    model_path = Path(args.model_path).as_posix()
    # 空加载模型, 可以几乎免去对存储空间的占用, 只记录每层有几个参数, 而不实际去申请内存初始化这些参数, 在加载大模型时有很多好处, 比如这里用来计算模型存储空间的占用, 毕竟加载一次大模型还是挺费时间的~
    with init_empty_weights():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto"
        )  # 这里auto时会加载bfloat16格式，占用和float16一致
    total_size, largest_layer = calculate_maximum_sizes(model)
    data = []

    for dtype in ["float32", "float16", "int8", "int4"]:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        if dtype == "float32":
            dtype_total_size *= 2
            dtype_largest_layer *= 2
        elif dtype == "float16":
            pass
        elif dtype == "int8":
            dtype_total_size /= 2
            dtype_largest_layer /= 2
        elif dtype == "int4":
            dtype_total_size /= 4
            dtype_largest_layer /= 4
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


def _get_args() -> Namespace:
    """命令行参数解析为命名空间(可以看作可以用.来访问的字典)"""
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
    """延迟加载模型以达到快速显示页面的目的

    延迟加载需要用到多线程,主线程执行web页面的时候,用子线程去加载模型,只需要记录好模型的引用对象即可.
    (利用延迟加载,主线程中不加载而是放到子线程中,这样而等到页面渲染好,
    用户输入完提问后,取出模型做推理时,子线程已经加载好模型.)
    """

    def __init__(self, args):
        self.args = args
        self.model = None
        self.proc = None
        self.lock = threading.Lock()

    def _load_model(self) -> None:
        """加载模型和processor"""
        with self.lock:
            if self.model is None:  # 确保模型只加载一次
                print(f"Loading model: {self.args.model_path}")
                try:
                    model, proc = self._load_model_processor()
                    # model不一定是存有dtype变量的nn.Module类,
                    # 因此可以用这个函数来快速获取里面第一个参数的dtype。
                    dtype = get_first_parameter_dtype(model)
                except Exception:
                    self.lock.release()
                    import traceback

                    traceback.print_exc()
                    exit(-1)
                self.model = model
                self.proc = proc
                print(f"Model {self.args.model_path} loaded")
                print(f"{model.device=} model.dtype={dtype}")

    def _load_model_processor(
        self,
    ) -> tuple[Qwen2VLForConditionalGeneration, Qwen2VLProcessor]:
        """Qwen2-vl 加载模型时需要加载两个东西:
        1. 模型, 对应Qwen2VLForConditionalGeneration类
        2. processor(一个对图片和文本进行处理,转换为模型输入的预处理工具),对应AutoProcessor类

        借助from_pretrained方法,我们可以在加载模型,预处理器时自动处理某些步骤(比如一般加载模型的流程是:初始化->从文件中加载权重并复制到初始化后的类中)而直接返回结果.
        """
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
                "int8": "auto",  # 不提供量化，自己改吧
            }[args.dtype]
            if args.dtype != "auto"  # auto会采用config中的配置
            else args.dtype
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            # 支持: eager/flash_attention_2/sdpa
            attn_implementation=use_fa2,
            # auto: 平均分配到每个 GPU.
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        return model, processor

    def get_model(self) -> Qwen2VLForConditionalGeneration:
        """获取加载的模型，若尚未加载则触发加载"""
        if self.model is None:
            threading.Thread(target=self._load_model).start()
        return self.model

    def get_processor(self) -> Qwen2VLProcessor:
        """获取加载的processor"""
        if self.proc is None:
            threading.Thread(target=self._load_model).start()
        return self.proc


def _transform_messages(
    messages: List[List[str | Tuple[str, ...]]],
    video_extensions=VIDEO_EXTENSIONS,
    image_extensions=IMAGE_EXTENSIONS,
    user_tag="user",
    assistant_tag="assistant",
) -> List[Dict[str, Any]]:
    """gradio的messages格式与qwen2的conversation不一致,需要转换

    模型的问答是按轮次来划分的:
        第一轮: <提问>-><回答>-> 第二轮: <提问>-><回答>-> ...
        (即便是加入文件,也是放在提问里面.)
    具体来说:
    1. gradio目前有多种`对话`的处理格式, 本代码中采用的格式为:
        [
            [(<文件1>,<文件2>, ...), None], # 如果传入文件,那么没有对应回答,如果这一行是文件,那么下一行跟用户提问
            [<提问>, <回答>],  # 注意,和上面的区别是提问是一个字符串,而上一行同样位置是一个存储文件的tuple.
            [("xxx1.jpg","xxx2.jpg"), None],
            ["描述下这两张图片", "这张图片xxx"],
            ...
        ]
    2. Qwen中的格式采用:
        [
            # 这里角色可以包括: system, user, assistant, 内容则是对应角色的提问或回答.
            {"role":<角色>, "content":<内容>},
            # 针对图片和视频的传输, Qwen2-vl 在 user 的 <内容> 部分会进一步处理, 因此我们可以将这两类文件放到其 <内容> 中:
            {"role":"user", "content":"你是谁?"},  # 纯文字
            {"role":"user", "content":[{"type":"image", "image": "xxx.jpg"}, {"type":"text", "text": "这张图里有什么?"}]},  # 图片+文字
            {"role":"user", "content":[{"type":"video", "video": "xxx.mp4"}, {"type":"text", "text": "这个视频讲了什么?"}]},  # 视频+文字
            ...
        ]
        (值得注意的是, 对视频或图片的token限制也可以加在content里面. 可以参考下面的处理)
    3. 发现了吗,上面两种对话格式不统一,因此送入模型的预处理器前还需要做一次处理,将gradio格式转为qwen预处理支持的格式.而gradio中文件和提问是放在多个列表里的,对话轮次的切换仅通过回答是否是None来判断.
    """
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

        if message[1]:  # 如果回答里有值，说明当前轮对话完成，接下来做下一轮对话的处理。
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


# modify from https://github.com/gradio-app/gradio/blob/4e1f7dbcb2ea2a0cc29bb76faf5758a9f4afcd6d/demo/chatbot_examples/run.py#L1, 参考这里可以看到gradio给出的带文件传输的chatbot实现
def print_like_dislike(x: gr.LikeData) -> None:
    print(f"{x.index=} {x.value=}{x.liked=}")


def add_message(
    history: List[List[str | Tuple[str, ...]]], message: Dict
) -> tuple[List[List[str | Tuple[str, ...]]], gr.MultimodalTextbox]:
    """
    Params:
        history: gradio的一种对话格式, 可以参考 `_transform_messages` 的文档注释.
        message: gr.MultimodalTextbox类, 可以当作字典访问,里面有file和text,分别表示提供的文件和提问.
    """
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append(
            (message["text"], None)
        )  # 这里填空是因为还需要把history数据转换后给模型进行回复，然后才能赋值到这里。
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def _pred(
    messages: List[List[str | Tuple[str, ...]]],
    temperature: float,
    topk: int,
    topp: float,
    processor: Qwen2VLProcessor,
    model: Qwen2VLForConditionalGeneration,
):
    """模型对话的主要逻辑, 这段代码参考了Qwen2-vl官方的 web demo的一部分流程.
    先转换出qwen2-vl需要的格式
    然后将文本和图像/视频分别送入预处理器(在此之前,图像/视频要借助官方提供的process_vision_info函数resize为28*28的倍数)
    然后送入模型进行推理,模型推理的结果作为回答."""
    messages = _transform_messages(messages)

    # 这里首先把messages对话格式转为纯文本的特殊格式
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 这里对图片/视频做resize处理，主要是模型内视觉层对图片的宽高有特定限制。
    image_inputs, video_inputs = process_vision_info(messages)
    # 开始通过预处理器， 将文本和图片/视频作为输入， 处理出模型需要的数据: token_id列表 和 特定形状的一堆像素点
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)  # 结果送入模型所在的设备（CPU或某GPU卡）

    streamer = TextIteratorStreamer(
        processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )  # 借助TextIteratorStreamer可以提供一个流式的接口，是的模型每生成一个token就返回这个token对应的文本。

    # 模型在生成token前有一个后处理，这里简单介绍贪心解码和采样解码：
    # 当使用贪心解码时，设置do_sample = False, 对于下一个token，模型总会选择预测的概率最大的那个。
    # 当使用采样时，字如其名，就是随机的选择。首先会对输出的下一个token的概率分布做一些简单变换（比如temperature越大，可以让概率分布越平均， topK和topP则减小待采样的词表），然后对剩余的词表进行加权的随机选择（因为加权，所以概率大的还是有大的机率被选中，但是如果temperature设置过大，反而把剩余所有词表的概率平均化了，这样大家的权重都接近1:1）
    # 因此也可以说，temperature控制模型的创造性，越大，模型采样到不同词的可能越大，模型的回答便越发散。
    _gen_kwargs = (
        dict(temperature=temperature, top_p=topp, top_k=topk)
        if temperature
        else dict(do_sample=False)
    )
    # max_new_tokens主要限制模型回答的最大token长度，当超过这个token就会停止。
    gen_config = GenerationConfig(max_new_tokens=512, **_gen_kwargs)

    # 使用子线程启动模型的推理，结果会自动添加到streamer接口中。
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


def bot(
    history: List[List[str | Tuple[str, ...]]],
    temperature: float,
    topk: int,
    topp: float,
) -> Generator[List[List[str | Tuple[str, ...]]], Any, None]:
    """这里是输入提问并点击提交后触发回答的逻辑"""
    _gc()  # 可以清除一下上一次回答的存储碎片
    # 然后将提问与之前轮次的对话送入_pred让模型针对这些上文进行推理
    model, proc = loader.get_model(), loader.get_processor()
    # 这里会返回一个流式的接口,通过for循环即可获取接口里新添加进去的回答,然后拼接到history里流式的返回给gradio即可.
    stream = _pred(history, temperature, topk, topp, processor=proc, model=model)
    history[-1][1] = ""
    for it in stream:
        history[-1][1] += it
        yield history


def web_demo(args: Namespace):
    """创建gradio应用程序"""
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
                # 一个隐藏的选项，可以控制 Temperature、top p、top k
                temperature = gr.Slider(0.0, 1.0, step=0.01, label="Temperature")
                topk = gr.Slider(-1, 1000, step=2, label="Top K")  # need?
                topp = gr.Slider(0.0, 1.0, step=0.01, label="Top P")  # need?
        # 多模态的输入会先调用 add_message，然后调用 bot，最后清除输入框中的内容（因为已经显示在chatbot里了）
        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(
            bot, [chatbot, temperature, topk, topp], chatbot, api_name="bot_response"
        )
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        # 这里主要是给chatbot的每个回答绑定一个用户偏好反馈的结果打印
        chatbot.like(print_like_dislike, None, None)
    demo.launch(max_threads=2, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    args = _get_args()
    # 在这里检测flash-attn是否安装和启用
    print("flash-attn 已安装" if is_flash_attn_2_available() else "flash-attn 未安装")
    print(
        "flash-attn 已启用。"
        if args.flash_attn2 and is_flash_attn_2_available()
        else "flash-attn 未启用。"
    )
    cal_model_size(args)  # 在每次启动模型时会先显示模型占用
    if args.cal_size is False:
        loader = LazyModelLoader(args)
        loader.get_model()  # 在这里手动提前触发一下模型加载
        web_demo(args)  # 运行web demo
