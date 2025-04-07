from flask import Flask, request, jsonify, render_template, session
import torch
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import gc
import re
import uuid

app = Flask(__name__)
app.secret_key = "llama4-scout-chatbot-secret-key"  # 用于session加密

# 全局变量存储预加载的模型和tokenizer
MODEL_ID = "/pfs/mt-euDpOR/nlp/personal/shufan.jiang/models/LLM-Research/Llama-4-Scout-17B-16E-Instruct"
tokenizer = None
model = None
# 用于存储对话历史的字典
chat_histories = {}
# 默认值设置
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_HISTORY_LENGTH = 10

# 在应用启动前预加载模型
def load_model():
    global tokenizer, model
    print("正在加载模型和tokenizer，请稍候...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 加载模型
    model = Llama4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("模型加载完成！")

def clean_response(text):
    """清理模型响应中的特殊标记"""
    # 根据截图中看到的标记，定义可能的标记形式
    patterns = [
        # 直接匹配具体的标记
        '<|eot|>',
    ]
    
    # 应用所有模式
    for pattern in patterns:
        text = text.replace(pattern, '')
    
    # 使用正则表达式处理可能的其他token
    text = re.sub(r'<[\|/]?eot[\|]?>', '', text)  # 匹配形如 <eot>, </eot>, <|eot|> 等
    
    return text.strip()

@app.route('/')
def home():
    # 创建会话ID
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
    
    # 如果是新会话，初始化聊天历史
    chat_id = session['chat_id']
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    
    return render_template('index.html', chat_id=chat_id)

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        # 确保模型已加载
        if tokenizer is None or model is None:
            return jsonify({"error": "模型正在加载中，请稍后再试"}), 503
        
        data = request.json
        user_input = data.get('user_input', '')
        chat_id = data.get('chat_id', session.get('chat_id', str(uuid.uuid4())))
        
        # 获取前端传递的参数，如果没有则使用默认值
        max_new_tokens = int(data.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS))
        max_history_length = int(data.get('max_history_length', DEFAULT_MAX_HISTORY_LENGTH))
        
        # 参数限制，确保在合理范围内
        max_new_tokens = max(256, min(max_new_tokens, 2048))
        max_history_length = max(2, min(max_history_length, 20))
        
        if not user_input:
            return jsonify({"error": "请输入问题"}), 400
        
        # 获取或初始化聊天历史
        if chat_id not in chat_histories:
            chat_histories[chat_id] = []
        
        # 添加用户消息到历史记录
        chat_histories[chat_id].append({"role": "user", "content": user_input})
        
        # 从历史记录构建消息列表，使用前端传递的历史长度
        messages = chat_histories[chat_id][-max_history_length*2:]  # 用户和助手消息各算一条
        
        # 应用chat模板
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        
        # 生成响应，使用前端传递的token数量
        with torch.no_grad():
            outputs = model.generate(**inputs.to(model.device), max_new_tokens=max_new_tokens)
        response = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 清理响应，移除结束标记
        cleaned_response = clean_response(response[0])
        
        # 添加模型回复到历史记录
        chat_histories[chat_id].append({"role": "assistant", "content": cleaned_response})
        
        # 如果历史记录太长，保留最新的max_history_length条
        if len(chat_histories[chat_id]) > max_history_length * 2:  # 用户和助手消息各占一半
            chat_histories[chat_id] = chat_histories[chat_id][-max_history_length*2:]
        
        return jsonify({
            "response": cleaned_response,
            "chat_id": chat_id,
            "max_new_tokens": max_new_tokens,
            "max_history_length": max_history_length
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        data = request.json
        chat_id = data.get('chat_id', session.get('chat_id'))
        
        if chat_id and chat_id in chat_histories:
            chat_histories[chat_id] = []
            return jsonify({"success": True, "message": "聊天历史已清除"})
        else:
            return jsonify({"success": False, "error": "无效的会话ID"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # 在另一个线程中预加载模型
    import threading
    threading.Thread(target=load_model).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) 