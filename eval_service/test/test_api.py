system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to test any python code, writing it inside ```python and ``` tags following with "```output". Put your final answer within \\boxed{}.:
"""

math_problem = """Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
"""

from openai import OpenAI
client = OpenAI(api_key="sk-proj-1234567890", base_url="http://0.0.0.0:5000") # Replace with your local server address

completion = client.chat.completions.create(
    model="VerlTool/acecoder-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-250-step",
    messages=[
		{
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": math_problem
        }
    ],
    temperature=0,
    max_tokens=2048,
    top_p=1,
    n=1,
)
print(completion.choices[0].message.content)

tempalte = """<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{math_problem}
<|im_end|>
<|im_start|>assistant
"""
completion = client.completions.create(
    model="VerlTool/acecoder-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-250-step",
    prompt=tempalte.format(system_prompt=system_prompt, math_problem=math_problem),
    temperature=0,
    max_tokens=2048,
    top_p=1,
    n=1,
)
print(completion.choices[0].text)


completion = client.completions.create(
    model="VerlTool/acecoder-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-69k-sys3-250-step",
    prompt=f"system\n{system_prompt}\n\nuser\n{math_problem}\nassistant\n",
    temperature=0,
    max_tokens=1241241,
    top_p=1,
    n=1,
)
print(completion.choices[0].text)