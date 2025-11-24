# How to Train a "Me"

This project will use "me" as a prototype and use a special dataset to fine-tune the large language model, dedicated to creating an AI digital human that can truly reflect my personality traits - including but not limited to my tone, expression, and thinking patterns, etc. Therefore, whether it is daily chatting or sharing feelings, it communicates in a familiar and comfortable way, as if I were by their side. The entire process is transferable and replicable, and the highlight is the production of the dataset.

> Project Background Details
>
> Due to the accelerated pace of life nowadays, many people find it difficult to accompany their families and friends often due to busy work or geographical distance, leading to an intensified sense of emotional alienation. In this era of rapid digital development, people are increasingly eager to bridge the gap in time and space through technical means. Therefore, AI emotional companionship products have emerged. However, despite the huge market demand, existing AI emotional companionship services still have many shortcomings. One is the high cost of customized services, which is difficult to popularize; another is that information security is difficult to guarantee; and another is the lack of personification, which makes it easy for people to feel out of place. After comprehensive consideration, we believe that fine-tuning on role-playing is a feasible path. Because fine-tuning is re-learning based on a pre-trained model, the model itself already has strong text understanding ability, logical ability, generalization ability, etc., and only needs some data additions of specific character personality styles to play the role we want well.

*Okay~ Next, let's start an immersive experience of the complete training process of a warm-hearted AI~*

---

Note that this demonstration is based on the github open source project "MemoTrace" ([https://github.com/LC044/WeChatMsg]), iFLYTEK Spark MaaS platform ([https://training.xfyun.cn/modelSquare])

## Dataset Production

First, we need to obtain the raw data, which is the JSON format of WeChat chat records. First log in to WeChat on the computer, synchronize chat information, then we enter the github project page to download the "MemoTrace" related files to the local computer, and then run the MemoTrace.exe file to activate the application. Select the chat record with a friend, export the chat content to JSON format and save it to the computer (due to time considerations, I will not go into details, everyone can follow the project readme document), and then you can open it in VSCode to view. The content includes three fields: conversations, role, and content. In order to facilitate the model to learn better, we need to "refine" it. You can refer to the following general idea of data processing:

```python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy
# Standardize role naming
def convert_to_sharegpt_format(original_data,new_system_value=None):
    sharegpt_data = []
    for conversation in original_data:
        new_conversation = {
            "conversations": [],
            "system":  new_system_value,
            "tools": "[]"  # If there is no tool call, it can be left empty or set to an empty list
        }
        system_message = None
        for msg in conversation["conversations"]:
            # if msg["role"] == "system":
            #     system_message = msg["content"]
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
                
# If a message with "role": "system" already exists in the original data, the content of this message will be prioritized for setting the "system" field of the new conversation. Therefore, even if you pass the new_system_value parameter when calling convert_to_sharegpt_format, once a system message defined in the original data is encountered, it will overwrite the new value you passed in.

        # # Set system message to system field
        # if system_message:
        #     new_conversation["system"] = system_message
        sharegpt_data.append(new_conversation)
    return sharegpt_data
 
# Read original JSON data
with open('z.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Convert to ShareGPT format

# Batch modify the value of "system" to "New System Value"
# Add optimized system prompt words, which can be understood as persona prerequisites
new_system_value = "You are (replace with protagonist name, persona), for you, the needs and feelings of others come before yourself. In addition, you like to listen very much..."

sharegpt_formatted_data = convert_to_sharegpt_format(original_data, new_system_value)

# Write to new JSON file

with open('sharegpt_formatted_data.json', 'w', encoding='utf-8') as f:
    json.dump(sharegpt_formatted_data, f, ensure_ascii=False, indent=2)

print("Data conversion completed and saved as sharegpt_formatted_data.json")
```

Here we use the ShareGPT format mainly because it is suitable for multi-turn dialogue model training, and is more suitable for role-playing scenarios than other formats. Secondly, it standardizes role naming, which facilitates model understanding and avoids confusion and hallucinations to a certain extent. Another one is the perfection and supplementation of the persona. And the most important point, be sure to remember to desensitize sensitive information in the dataset (including phone numbers, passwords, addresses, etc.)!!!!!~~

Remember to change the value of the system prompt word (system), file save location, etc.!!!

*For this dataset, there are two special points that everyone can refer to for retention and optimization. One is that I kept all the text converted from kaomoji and emoji packages (such as [grievance], etc.), because experiments have proven that this helps the large model understand the emotional color characteristics of the discourse and can imitate personalized expressions. We found that after fine-tuning, it can also return kaomoji and emoji packages to the user in a timely manner according to the context. The other is that I rewrote the MBTI dataset, adapting the thinking dataset of the personality corresponding to myself into the format of chat information, and changing some representative problem scenarios into dialogues and adding them to the dataset, which will make it easier for the model to capture the characteristics of the character's thinking mode.*
MBTI dataset github address ([https://huggingface.co/datasets/pandalla/Machine_Mindset_MBTI_dataset])

Another thing is to pay attention to filtering and processing incomplete dialogue fragments, including but not limited to supplementing dialogues, deleting some fragments with topic jumps, etc. We have about 3000 pieces of data. Considering that my topics are often quite jumpy and I don't answer questions consecutively, I manually cleaned and processed them again. Some fragments with serious topic jumps bring great difficulty to the model's understanding, so some were manually removed. For some dialogues with jumps but value, we tried to add appropriate background information to help the model establish the correct context.

Of course, for refinement, everyone can continue to explore, such as accessing the API for dialogue emotional color judgment, adding emotional labels, feature labels describing the character's personality traits, professional background, hobbies, etc., which are all conducive to the humanization and personalization of AI digital humans. Examples are as follows.

```json
{ "from": "gpt", "value": "Learned it?[Laugh]?", "character_traits": { "personality": "Smart, enthusiastic, kind", "occupation": "Programmer", "interests": ["Programming", "Reading", "Travel"] } }

```

> [!NOTE] Dataset Description
> Why do we choose chat records as the dataset? The reason is simple. Our daily communication can gradually depict a person's appearance in our minds, whether kind or innocent and cute, all of which are constantly revealed through many details in our daily communication with friends and family. So asking the model to learn a person and imitate a person, isn't it honing in these details? This is undoubtedly the best template for creating a soul clone, so we choose to work hard on chat scenarios and content~~


Okay, that's it for the dataset for now. Next, go to the iFLYTEK Spark MaaS platform for fine-tuning.

## Model Fine-tuning Training
In this step, we use the iFLYTEK Spark MaaS platform ([https://training.xfyun.cn/modelSquare]). Everyone can choose according to their own needs. In fact, the main thing is to need an open source general large model, and then give it the dataset for training and learning (fine-tuning). Here you can refer to the Datawhale official tutorial [Customize your exclusive large model from scratch ([https://www.datawhale.cn/activity/110?subactivity=21])] for the usage tutorial, so I won't go into details here~
But note one point, according to our steps above, our dataset belongs to the ShareGPT format, don't choose the wrong one~

>
> First of all, some suggestions on the setting of some parameters are provided for reference: Learning Rate and Epochs are two key hyperparameters that have a significant impact on the final model performance. Choosing the appropriate learning rate and number of training times requires comprehensive consideration of specific tasks, dataset characteristics, and computing resources. A learning rate that is too large may cause the model to fail to converge, while a learning rate that is too small may cause the training process to be slow or even fall into a local optimal solution. Therefore, a smaller learning rate is usually used in the fine-tuning stage to avoid destroying the useful information already learned. You can understand this parameter as the magnitude of its change in its own cognition based on the learning content. It is recommended to start small, which will largely ensure that the loss function curve is gentle and not too oscillating. The number of training times is the number of times the entire training set is traversed. Fine-tuning usually does not require too many training cycles because the pre-trained model already has a certain knowledge base. Also, if the dataset itself is not large (less than 500 items), too many training cycles may lead to overfitting, that is, the model performs well on the training set but poorly on unseen data. For these two parameters, you can continuously adjust them in combination with the training Loss curve. The purpose is to prevent the curve from oscillating too much, nor can it be too flat. Too flat may mean overfitting~ Then there is the temperature coefficient. My experimental results show that 0.9 is better. Everyone can also debug more according to the response effect.
> Secondly, the choice of fine-tuning method mainly considers Lora and full fine-tuning. Why did I finally choose Lora instead of full fine-tuning? We comprehensively considered time and resource costs. We adopted LoRa fine-tuning. When the dataset scale is small, there is actually no need to comprehensively adjust all parameters, because most pre-trained models already have good initialization and feature extraction capabilities. In this case, adopting full fine-tuning not only increases unnecessary computational burden, but may also lead to a lengthy and inefficient training process. Of course, if the dataset is very large (tens of thousands), that's another story.

A relatively good fine-tuning effect will look something like this:
![](/images/图片1.png)

![](/images/图片2.png)

(The prototype inside is me. I answer that I like kaomoji and various emoji packages, which will make the other party feel very warm. I also care more about the emotional expression of some punctuation marks. You can see that I have a special liking for exclamation marks and tildes hahaha. Overall, the effect is quite good~)

## Frontend Page Display
This step mainly requires calling the API of your fine-tuned model. We made a relatively beautiful chat interface display. Everyone can imitate and explore it. It is quite interesting and tests your aesthetics. Here mainly shows the relevant code for API calling. Our API call connects to the fine-tuned model's API via *WebSocket*:

```javascript
            async function handleSendMessage() {

                const message = userInput.value.trim();

                if (!message) return;

  

                addMessage('User', message, 'user-message');

                userInput.value = '';

  
# Here is the address of the model fine-tuned and published in the iFLYTEK platform, change it to your own!!
                const ws = new WebSocket('wss://maas-api.cn-huabei-XX');

  
# The following lines are API calls, in the "Information Call" at the bottom right of the "Service Control" page
                ws.onopen = () => {

                    const requestData = {
# This needs to be changed to your own relevant API!!
                        header: {

                            app_id: "XXXXXXXX",

                            uid: "XXXXX",

                            patch_id: ["XXXXXXXXXXXXXXXXXXX"]

                        },

                        parameter: {

                            chat: {
# Here we chose the spark13b model for fine-tuning. If you are not using this one, remember to check the API call documentation of the iFLYTEK platform and change it accordingly!!
                                domain: "xspark13b6k",
# This is the temperature coefficient
                                temperature: 0.9

                            },

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

  

                let fullResponse = ''; // Used to store the complete AI response

  

                ws.onmessage = (event) => {

                    const response = JSON.parse(event.data);

                    if (response.header.code === 0) {

                        // Concatenate the content received each time

                        const aiResponsePart = response.payload.choices.text.map(choice => choice.content).join('');

                        fullResponse += aiResponsePart;

  

                        // Check if it is the last response

                        if (response.payload.choices.status === 2) {

                            addMessage('AI', fullResponse, 'ai-message');

                        }

                    } else {

                        console.error('Error:', response.header.message);

                        addMessage('AI', 'Sorry, a server error occurred, please try again later.', 'ai-message');

                    }

                };

  

                ws.onerror = (error) => {

                    console.error('WebSocket Error:', error);

                    addMessage('AI', 'Sorry, a connection error occurred, please try again later.', 'ai-message');

                };

  

                ws.onclose = () => {

                    console.log('WebSocket connection closed');

                };

            }

```

This code defines an asynchronous function named `handleSendMessage`, which is called when the user clicks the send button. The function first gets the message entered by the user, and then connects to the specified API endpoint via ___WebSocket___. After the connection is successful, it sends a JSON request data containing the user's message. Then, it listens to the `onmessage` event of the WebSocket, receives and processes the response returned by the API, concatenates the response content, and displays it in the chat interface after receiving the complete response. If an error occurs or the connection is closed, it handles these situations accordingly and displays an error message in the chat interface~~


After success, the effect will be roughly like this:
![](屏幕截图 2025-01-23 010103.png)
Now you can tease him/her everyday when you have nothing to do~

Okay, if you finished this step, congratulations!!!! You already have the soul clone of the person you dream of. It should allow you or other users to experience warm companionship immersively!!! Of course, this can also be expanded. Mainly this idea can be migrated to other application scenarios, such as anime characters that may not exist in the world you like, or relatives who have long left you. All are fine. As long as it is related to emotional companionship, you can actually try it. *Our ultimate goal is to reduce costs so that more people can enjoy this form of warm companionship, and then explore how not to let people feel out of place in this process. Think about it, if you pour your heart out to it, and due to some sensitive word avoidance settings, it replies "Sorry, I am just a large language model", that really makes people's hearts cold~*

Subsequently, I will also explore the addition of voice functions. Friends who are interested are welcome to communicate~~

