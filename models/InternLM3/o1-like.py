import groq
import time
import os
import json

client = groq.Groq()

def make_api_call(messages, max_tokens, is_final_answer=False, custom_client=None):
    global client
    if custom_client != None:
        client = custom_client
    
    for attempt in range(3):
        try:
            if is_final_answer:
                response = client.chat.completions.create(
                    model="internlm3-8b-instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
            ) 
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model="internlm3-8b-instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt, custom_client=None):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300, custom_client=custom_client)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 25: # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
            break
        
        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice."})
    
    start_time = time.time()
    final_data = make_api_call(messages, 1200, is_final_answer=True, custom_client=custom_client)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data, thinking_time))

    yield steps, total_thinking_time
