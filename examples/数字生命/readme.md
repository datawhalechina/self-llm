# 如何练就一个我
本项目将以我为原型，利用特制的数据集对大语言模型进行微调，致力于创造一个能够真正反映我的个性特征的AI数字人——包括但不限于我的语气、表达方式和思维模式等等，因此无论是日常聊天还是分享心情，它都以一种既熟悉又舒适的方式交流，仿佛我在他们身边一样。整个流程是可迁移复制的，亮点是数据集的制作。

> 项目背景详情
>
> 由于现下生活节奏加快，许多人因工作繁忙或地理距离而难以时常陪伴家人与朋友，导致情感上的疏离感加剧。在这个数字化快速发展的时代，人们越来越渴望通过技术手段弥补时间和空间上的隔阂。因而AI情感陪伴类产品应运而生，然而，尽管市场需求巨大，现有的AI情感陪伴服务仍存在诸多不足。一个是定制化服务成本高昂，难以普及，一个是信息安全难以保障，还有一个是人格化缺失，容易让人出戏。综合考量下，我们认为在角色扮演上微调不失为一个可行的路子。因为微调是在预训练模型基础上再学习，模型本身已经有较强的文本理解能力、逻辑能力、泛化能力等，只需要一些特定人物个性风格的数据加成就能很好的扮演我们想要的角色了。

*好的~接下来就让我们开始沉浸式体验一下一个暖心AI的完整训练流程吧~*

---

注意，本次演示，是基于github开源项目“留痕”（[https://github.com/LC044/WeChatMsg]）、 讯飞星辰MaaS平台([https://training.xfyun.cn/modelSquare])

## 数据集的制作
首先我们需要获取原始数据，即微信聊天记录的JSON格式。先在电脑端登录微信，同步聊天信息，接着我们进github项目页把“留痕”相关的文件下载到电脑本地，然后运行MemoTrace.exe文件，把应用激活。选择和某个朋友的聊天记录，将聊天内容导出为JSON格式保存至电脑（由于时间考虑，我就不赘述了，大家可以按照项目自述文档操作），结束可以在VSCode中打开查看，内容包括conversations、role和content三个字段，为了便于模型更好地学习，我们需要再“精修”一下，可参考以下数据处理的大致思路:

```
# -*- coding: utf-8 -*-

import json
from copy import deepcopy
#标准化角色命名    
def convert_to_sharegpt_format(original_data,new_system_value=None):
    sharegpt_data = []
    for conversation in original_data:
        new_conversation = {
            "conversations": [],
            "system":  new_system_value,
            "tools": "[]"  # 如果没有工具调用，可以留空或设置为空列表
        }      
        system_message = None
        for msg in conversation["conversations"]:
            # if msg["role"] == "system":
            #     system_message = msg["content"]
            if msg["role"] == "user":
                new_conversation["conversations"].append({
                    "from": "human",
                    "value": clean_content(msg["content"])
                })
            elif msg["role"] == "assistant":
                new_conversation["conversations"].append({
                    "from": "gpt",
                    "value": clean_content(msg["content"])
                })
                
#如果原始数据中已经存在"role": "system"的消息，那么这个消息的内容会被优先用于设置新对话的"system"字段。因此，即使你在调用convert_to_sharegpt_format时传递了new_system_value参数，一旦遇到原始数据中定义的系统消息，它就会覆盖你传入的新值。      

        # # 将系统消息设置为system字段
        # if system_message:
        #     new_conversation["system"] = system_message
        sharegpt_data.append(new_conversation)
    return sharegpt_data
 
# 读取原始JSON数据
with open('z.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 转换为ShareGPT格式

# 批量修改"system"的值为"New System Value"
#添加优化系统提示词，可以理解为人设前提
new_system_value = "你是（替换为主角名字，人设），对你来说，他人的需求感受在自己之前。此外，你很喜欢倾听......"

sharegpt_formatted_data = convert_to_sharegpt_format(original_data, new_system_value)

# 写入新的JSON文件

with open('sharegpt_formatted_data.json', 'w', encoding='utf-8') as f:
    json.dump(sharegpt_formatted_data, f, ensure_ascii=False, indent=2)

print("数据转换完成并保存为 sharegpt_formatted_data.json")
```

这里我们用ShareGPT的格式主要是它适用于多轮对话的模型训练，相较于别的格式更适用于角色扮演的场景，其次就是规范了角色命名，便于模型的理解，一定程度避免混淆幻觉，还有一个就是人设的完善补充，还有最重要的一点，一定一定要记得对数据集进行敏感信息脱敏（包括电话，密码，地址等）！！！！！~~

记得改系统提示词的值（system），文件保存位置等！！！

*对于这个数据集，有两个特别的点，大家可参考保留优化，一个就是我保留所有的颜文字、表情包转的文字（如[委屈]等），因为经过实验证明，这有助于大模型理解话语的情感色彩特征，并能模仿到个性化的表达方式，我们发现微调后它能根据语境适时也配上颜文字和表情包返回给用户。一个就是我改写了MBTI的数据集，将我自己对应的那个人格的思维数据集按照聊天信息的形式格式改编，把一些比较有代表性的问题场景改成对话加到数据集里，这将更便于模型捕捉角色思维模式的特征。*
MBTI数据集github地址（[[https://huggingface.co/datasets/pandalla/Machine_Mindset_MBTI_dataset`]]）

再有就是要注意对对话片段内容不完整的进行筛选处理，包括但不限于补充对话、删除一些话题跳跃的片段等。我们大概有3000条数据，考虑到我很多时候话题都比较跳跃，没连着回答问题，所以我是人为再清洗处理。有些话题跳跃严重的片段给模型带来较大的理解难度，一些人为去除，对于某些存在跳跃但有价值的对话，我们尝试添加适当的背景信息，帮助模型建立正确的语境。

当然对于精修，大家可以继续探索，比如接入对话情绪色彩判断的API，添加情感标签、描述角色的性格特点、职业背景、兴趣爱好等的特征标签，都有利于AI数字人的人性化、个性化，示例如下。

```
{ "from": "gpt", "value": "学会了吗[笑]？", "character_traits": { "personality": "聪明、热情、善良", "occupation": "程序员", "interests": ["编程", "阅读", "旅行"] } }

```

> [!NOTE] 数据集说明
> 为什么我们选择聊天记录来做数据集，原因无他，我们日常的交流就能在脑海中渐渐描摹出一个人的模样，慈眉善目的，或是天真可爱的，都通过我们和朋友家人的日常交流中的很多细节不断在显现，那让模型学一个人模仿一个人，不就是在这些细节里磨炼吗，这无疑是造就一个灵魂分身最好的模版，所以我们选择在聊天场景、内容中下功夫~~


好了，那数据集就先告一段落，接下来就到讯飞星辰MaaS平台进行微调.

## 模型微调训练
这一步，我们使用的是讯飞星辰MaaS平台([https://training.xfyun.cn/modelSquare])，大家可以根据自己的需要自行选择,其实主要就是需要开源的通用大模型，然后把数据集给它进行训练学习（微调），这里使用教程可以看看Datawhale的官方教程【零基础定制你的专属大模型（[https://www.datawhale.cn/activity/110?subactivity=21]）】作为参考，这里就不赘述了~
不过要注意一点，按照我们上面的步骤，我们的数据集属于ShareGPT格式，别选错了~

> 
> 首先是对一些参数的设置提一些建议以供参考：学习率（Learning Rate）和训练次数（Epochs）是两个关键的超参数，它们对最终模型性能有着重要影响。选择合适的学习率和训练次数需要结合具体任务、数据集特征以及计算资源来综合考虑。一个过大的学习率可能会导致模型无法收敛，而太小的学习率则可能导致训练过程缓慢甚至陷入局部最优解。因此，在微调阶段通常使用较小的学习率，以避免破坏已经学到的有用信息 。可以将这个参数理解为它依据学习内容去改变它自己认知的幅度的大小，建议是从小开始，这样会较大程度保证损失函数曲线的平缓，不太震荡。训练次数即整个训练集被遍历的次数。微调通常不需要太多的训练周期，因为预训练的模型已经具备了一定的知识基础。还有，如果数据集本身不大（少于500条），过多的训练周期可能会导致过拟合，即模型在训练集上表现很好，但在未见过的数据上表现不佳。对于这两个参数，可以结合训练的Loss曲线进行不断地调整，目的就是让曲线不要过于震荡，也不能太平，太平可能过拟合~然后就是温度系数，我的实验结果显示0.9为佳，大家也可以根据回应的效果多调试。
> 其次就是微调方式的选择，主要考虑Lora和全量精调，那为什么我最终选择Lora,而不是全量精调，我们综合考虑时间和资源成本，我们采用LoRa微调，当数据集规模较小时，实际上没有必要对所有参数进行全面调整，因为大部分预训练模型已经具备了良好的初始化和特征提取能力。在这种情况下，采用全量精调不仅增加了不必要的计算负担，还可能导致训练过程变得冗长且低效。当然数据集很大的话另说（几万条）。

比较好的微调效果大概会像下面这样：
![](/images/图片1.png)

![](/images/图片2.png)

（里面的原型就是我，我回答喜欢颜文字和各种表情包，会让对方感觉很温馨，我也比较在意一些标点符号的情感表达，可以看到我是对感叹号波浪号等非常情有独钟的哈哈哈哈，整体来说，效果是比较不错的~）

## 前端页面展示
这一步主要是需要调用你微调好的模型的API，我们做了一个比较美观的聊天界面展示，大家可以模仿探索一下，还是比较有意思的，考验你的审美了。这里主要展示一下API调用的相关代码，我们的API调用是通过*WebSocket连接*到训练好的模型的API：

```
            async function handleSendMessage() {

                const message = userInput.value.trim();

                if (!message) return;

  

                addMessage('用户', message, 'user-message');

                userInput.value = '';

  
#这里是在讯飞平台里微调好并发布的模型的地址，改成你自己的！！
                const ws = new WebSocket('wss://maas-api.cn-huabei-XX');

  
#以下几行为API调用，在“服务管控”页的右下角“信息调用”处
                ws.onopen = () => {

                    const requestData = {
#这要改成你自己的相关API！！
                        header: {

                            app_id: "XXXXXXXX",

                            uid: "XXXXX",

                            patch_id: ["XXXXXXXXXXXXXXXXXXX"]

                        },

                        parameter: {

                            chat: {
#这里我们选用的事星火13b的模型进行微调，如果不是用的这个，记得看讯飞平台的API调用文档说明，对应的改！！
                                domain: "xspark13b6k",
#这是温度系数
                                temperature: 0.9

                            }

                        },

                        payload: {

                            message: {

                                text: [

                                    { "role": "user", "content": message }

                                ]

                            }

                        }

                    };

                    console.log(message)

                    ws.send(JSON.stringify(requestData));

                };

  

                let fullResponse = ''; // 用于存储完整的AI响应

  

                ws.onmessage = (event) => {

                    const response = JSON.parse(event.data);

                    if (response.header.code === 0) {

                        // 拼接每次接收到的内容

                        const aiResponsePart = response.payload.choices.text.map(choice => choice.content).join('');

                        fullResponse += aiResponsePart;

  

                        // 检查是否是最后一次响应

                        if (response.payload.choices.status === 2) {

                            addMessage('AI', fullResponse, 'ai-message');

                        }

                    } else {

                        console.error('Error:', response.header.message);

                        addMessage('AI', '抱歉，服务器出现错误，请稍后再试。', 'ai-message');

                    }

                };

  

                ws.onerror = (error) => {

                    console.error('WebSocket Error:', error);

                    addMessage('AI', '抱歉，连接出现错误，请稍后再试。', 'ai-message');

                };

  

                ws.onclose = () => {

                    console.log('WebSocket connection closed');

                };

            }

```

这段代码定义了一个名为`handleSendMessage`的异步函数，它在用户点击发送按钮时被调用。函数首先获取用户输入的消息，然后通过 ___WebSocket连接___ 到指定的API端点。在连接成功后，它发送一个包含用户消息的JSON请求数据。接着，它监听WebSocket的`onmessage`事件，接收并处理API返回的响应，将响应内容拼接起来，并在接收到完整响应后将其显示在聊天界面中。如果发生错误或连接关闭，它会相应地处理这些情况并在聊天界面中显示错误信息~~


成功后，效果大概会是下面这样：
![](屏幕截图 2025-01-23 010103.png)
那现在你就可以日常没事逗逗他/她啦~

好啦，如果你到这一步结束了，恭喜你！！！！已经拥有了你梦寐以求的那个人的灵魂分身了，它应该可以让你或者别的用户沉浸式体验有温度的陪伴啦！！！当然，这个还可以拓展，主要是这个思路，还能继续迁移到别的应用场景，比如你喜欢的世界上可能不存在的动漫角色，比如早已离你而去的亲人，都可以，只要是情感陪伴相关的，其实都可以尝试一下，*我们的终极目标就是降低成本，让更多的人可以享受到这种形式的暖心陪伴，然后探索怎么在这个过程中不让人出戏，想想，要是在你向它倾诉时，出于一些敏感词的规避设置，它回你一句对不起，我只是一个语言大模型，那真是让人心都凉了呢~*

后续，我还会探索语音功能的加成，有兴趣的朋友欢迎交流鸭~~

