import torch
import re
import time
import uuid
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional, Union
from config import ModelConfig, ToolConfig
from utils import extract_final_code, extract_python_tags, call_tool_server

class ModelService:
    """大模型推理与工具调用服务"""
    
    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        """初始化模型服务"""
        self.model_config = model_config
        self.tool_config = tool_config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """使用vllm加载模型"""
        print(f"正在加载模型: {self.model_config.model_path}...")
        self.model = LLM(
            model=self.model_config.model_path,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            max_model_len=self.model_config.max_model_len
        )
        
        
        print(f"正在加载模型的tokenizer...")
        self.tokenizer = self.model.get_tokenizer()
        
        
        print("模型加载完成")
        return self.model, self.tokenizer
    
    def format_system_user_prompt(self, system_prompt: str, user_message: str) -> str:
        """格式化系统提示和用户消息"""
        # 这里使用与训练时一致的格式
        
        # print(f"[DEBUG] system_prompt: {system_prompt}\nuser_msg: {user_message}")
        
        # return f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|>\n"
        
        return f"{system_prompt}\n{user_message}\n"
    
    def generate_with_tools_old(self, prompt: str) -> Dict[str, Any]:
        """使用工具调用生成回复"""
        if not self.model or not self.tokenizer:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 生成唯一的会话ID
        trajectory_id = str(uuid.uuid4())
        
        # 跟踪完整上下文
        full_context = prompt
        
        print(f"[DEBUG] passed_in prompt: {prompt}")
        
        generated_text = ""
        result_pieces = []
        
        # 轮次计数，用于判断是否满足最小工具调用次数
        turn_count = 0
        
        # 多轮工具调用循环:
        
        # record the context, initially it should be the passed-in prompt
        # while the total text generation length smaller than the limit:
        # 1. generate a chunk of text with maximum length: 1024
        # 2. in this chunk, loop through each character to see if this character match with (part of) the tool-calling tag: <python>.
        # if in the chunk of code none of the characters match with the tool-calling tag, then check if there are any final-output pattern (markdown python code block) in the generated text.
        # 2.1. if so, stop the generation and return the generated text.
        # 2.2. if part of the tool calling tag is found, then truncate everything after that first partially-generated tool-calling tag, add the missing part of the tool-calling tag, and then call the tool server to get the observation.
        # 5. append the observation to the generated text, treat the processed text as part of the new context. append it to the existing context. record the number of tokens, maintain the total token count parameter.
        # 6. continue to generate the next chunk of text. 
        
        for idx in range(self.tool_config.max_turns):
            print(f"执行第 {turn_count + 1} 轮推理...")
            
            # 调用模型生成下一段文本, 自然终止
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024
            )
            
            outputs = self.model.generate([full_context], sampling_params)
            output_text = outputs[0].outputs[0].text
            
            # 将当前生成的文本添加到结果中
            result_pieces.append(output_text)
            generated_text += output_text
            
            print(f"[DEBUG] current_output: {generated_text}")
            
            # 检查是否包含Python代码块
            python_code, has_tool_call = extract_python_tags(output_text)
            
            # 如果没有检测到工具调用
            if not has_tool_call:
                # 如果满足了最小调用次数要求或配置允许无工具调用停止，则结束
                if turn_count >= self.tool_config.min_action_num or not self.tool_config.no_action_as_stop:
                    print("未检测到工具调用，生成结束")
                    break
                    
                # 否则继续生成
                full_context += output_text
                continue
                
            # 执行工具调用
            print(f"检测到工具调用，执行Python代码...")
            tool_result = call_tool_server(
                self.tool_config.tool_server_url,
                trajectory_id,
                python_code,
                finish=False
            )
            
            # 增加轮次计数
            turn_count += 1
            
            # 更新上下文
            observation = tool_result["observation"]
            full_context = full_context + output_text + observation
            
            # 检查是否结束
            if tool_result["done"]:
                print("工具服务器指示生成结束")
                break
                
            # 检查是否已经有最终代码块
            if "```python" in generated_text and "```" in generated_text:
                print("检测到最终代码块，停止生成")
                break
        
        # 如果还需要生成最终输出
        if "```python" not in generated_text:
            print("生成最终代码块...")
            # 添加指示完成的提示
            final_prompt = full_context + "\n现在，请提供最终的代码解决方案：\n"
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024
            )
            
            outputs = self.model.generate([final_prompt], sampling_params)
            final_output = outputs[0].outputs[0].text
            result_pieces.append(final_output)
            generated_text += final_output
        
        # 提取最终代码
        final_code, has_final_code = extract_final_code(generated_text)
        
        return {
            "full_response": generated_text,
            "final_code": final_code,
            "reasoning_steps": result_pieces
        }
    
    def generate_with_tools(self, prompt, max_total_tokens=4096, max_chunk_tokens=1024):
        """
        Generate text with tool calls in a multi-turn loop.
        
        Args:
            prompt: Initial prompt for generation
            max_total_tokens: Maximum total tokens to generate
            max_chunk_tokens: Maximum tokens to generate in each chunk
            
        Returns:
            Generated text with tool interactions
        """
        # 初始化上下文为传入的提示词
        context = prompt
        total_tokens_used = len(self.tokenizer.encode(context))
        generated_text = ""
        action_step = 0
        
        # 定义工具调用标签
        tool_tags = [f"<{action}>" for action in self.tool_config.valid_actions]
        tool_stop_tags = [f"</{action}>" for action in self.tool_config.valid_actions]
        
        return_payload = {
            "full_response": "",
            "final_response": ""   
        }
        
        # 当总生成长度小于限制时，继续生成
        while total_tokens_used < max_total_tokens:
            action_step += 1
            print(f"Action step {action_step}")
            
            # 1. 生成一个长度最大为max_chunk_tokens的文本块
            inputs = self.tokenizer(context, return_tensors="pt", padding=True)
            
            # fix for qwen 
            # REF: https://github.com/vllm-project/vllm/issues/2947#issuecomment-1959569041
            sampling_params = SamplingParams(
                temperature=0,
                top_p=0.9,
                max_tokens=max_chunk_tokens,
                skip_special_tokens=False, 
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    [context],
                    sampling_params
                )
            
            response = outputs[0].outputs[0].text
            print(f"[DEBUG] 生成的文本: {response}")
            
            # 2. 处理生成的响应，检查工具调用
            has_action = False
            response = response.strip(' \n')
            
            # 如果配置设置为"无动作即停止"且已达到最小动作数
            if self.tool_config.no_action_as_stop and action_step >= self.tool_config.min_action_num:
                for stop_tag in tool_stop_tags:
                    if response.endswith(stop_tag):
                        # 找到完整的工具调用，截取到结束标签
                        has_action = True
                        response = response.split(stop_tag)[0] + stop_tag
                        break
            else:
                # 默认设置为找到动作
                has_action = True
                
                # 检查是否存在部分工具调用标签，如果存在则截断
                for tag in tool_tags:
                    if tag in response and not any(stop_tag in response for stop_tag in tool_stop_tags):
                        tag_start = response.find(tag)
                        tag_end = tag_start + len(tag)
                        
                        # 查找是否有完整的工具调用（开始和结束标签）
                        corresponding_stop_tag = tag.replace("<", "</")
                        stop_tag_pos = response[tag_end:].find(corresponding_stop_tag)
                        
                        if stop_tag_pos == -1:
                            # 只找到开始标签但没找到结束标签，截断到开始标签后
                            response = response[:tag_end]
                        else:
                            # 找到完整工具调用，保留到结束标签
                            response = response[:tag_end + stop_tag_pos + len(corresponding_stop_tag)]
            
            # 如果没有找到任何工具调用，检查是否有最终输出模式
            if not has_action:
                # 检查是否有markdown代码块等表示最终输出的模式
                if "```python" in response or "```" in response or action_step >= self.tool_config.max_turns:
                    
                    # match the pattern with the first occurrence
                    match_pattern = "\`\`\`python((.|\n)*?)\`\`\`"
                    
                    match = re.search(match_pattern, response)
                    if match:
                        # 提取匹配的代码块
                        final_code = match.group(0)
                        
                        # 将最终代码添加到返回值中
                        return_payload["final_response"] = final_code
                        return_payload["full_response"] = generated_text + response
                        
                    else:
                        # 如果没有找到匹配的模式，则返回完整响应
                        generated_text += response
                        return_payload["final_response"] = ""
                        return_payload["full_response"] = generated_text 
                    break
            
            # 3. 与工具服务器交互获取观察结果
            # 创建一个唯一ID用于此次交互
            traj_id = str(uuid.uuid4())
            
            # 与工具服务器交互
            next_obs, dones, valid_action = self.interact_with_tool_server(
                [traj_id], 
                [response], 
                [has_action], 
                torch.tensor([True])
            )
            
            # 如果交互完成，则停止生成
            if dones[0]:
                generated_text += response + next_obs[0]
                
                # TODO: fix return format
                
                break
            
            # 5. 将观察结果附加到生成的文本中，并更新上下文
            generated_text += response + next_obs[0]
            context = prompt + generated_text
            
            # 6. 更新已使用的令牌数量
            total_tokens_used = len(self.tokenizer.encode(context))
            
            print(f"Total tokens used: {total_tokens_used}/{max_total_tokens}")
            
            # 如果接近最大标记限制，则停止生成
            if total_tokens_used > max_total_tokens * 0.9:
                
                # TODO: fix return format
                
                break
        print(f"[DEBUG] 最终输出: {return_payload}")
        return return_payload
    
    def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """处理API请求并生成响应"""
        # 提取系统消息和用户消息
        system_prompt = "Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside <python> and </python> tags. The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between <python> and </python> tags are independent program. You can run Python code as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\nyour code here\n```. The final program will be evaluated against the test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty."
        user_message = None
        
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                if user_message is None:  # 仅使用第一条用户消息
                    user_message = message["content"]
        
        # 使用默认系统提示如果没有提供
        if not system_prompt:
            system_prompt = "Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside <python> and </python> tags. The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between <python> and </python> tags are independent program. You can run Python code as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\nyour code here\n```. The final program will be evaluated against the test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty."
        
        if not user_message:
            raise ValueError("未提供用户消息")
        
        # 格式化提示
        prompt = self.format_system_user_prompt(system_prompt, user_message)
        
        # 执行带工具调用的生成
        result = self.generate_with_tools(prompt)
        
        
        # 格式化为OpenAI API响应格式
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"{result['final_response']}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # 需要实际计算
                "completion_tokens": 0,  # 需要实际计算
                "total_tokens": 0  # 需要实际计算
            }
        }