import time
import uuid
import os
from vllm import LLM, SamplingParams
from typing import Dict, Any
from config import ModelConfig, ToolConfig
from utils import call_tool_server

class ModelService:
    """verl-tool model inference service"""
    
    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        """initialize model service"""
        self.model_config = model_config
        self.tool_config = tool_config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """load the model using VLLM backend"""
        print(f"Loading Model using VLLM: {self.model_config.model_path}...")
        available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        self.model = LLM(
            model=self.model_config.model_path,
            tensor_parallel_size=len(available_gpus),
            trust_remote_code=True,
            # max_model_len=self.model_config.max_model_len
        )
        
        print(f"Loading Tokenizer using VLLM: {self.model_config.model_path}...")
        self.tokenizer = self.model.get_tokenizer()
        
        
        print("Model loaded successfully.")
        return self.model, self.tokenizer
    
    def generate_with_tools(self, prompt, sampling_params):
        """
        Generate text with tool calls in a multi-turn loop.
        
        Args:
            prompt: Initial prompt for generation
            sampling_params: Sampling parameters for the model
            
        Returns:
            full_response: Generated text with tool interactions (not including the prompt)
        """
        context = prompt
        
        # keep trying to generate the response until reached the tool-calling limit
        for action_step in range(self.tool_config.max_turns):
            outputs = self.model.generate(
                [context],
                SamplingParams(**sampling_params)
            )
            response = outputs[0].outputs[0].text
            response = response.strip(' \n')
            
            print(f"action step {action_step} full response", context+response) # TODO: delete 
            
            if not response.endswith("```output"):
                return context + response
            else:
                traj_id = str(uuid.uuid4())
                tool_response = call_tool_server(
                    self.tool_config.tool_server_url,
                    traj_id, 
                    response, 
                )
                next_obs = tool_response["observation"]
                valid_action = tool_response["valid"]
                if valid_action:
                    context += response + next_obs
                else:
                    continue
                print(f"action step {action_step} full response after tool call", context) # TODO: delete 
        
        sampling_params["stop"].remove("```output")
        outputs = self.model.generate(
                    [context],
                    SamplingParams(**sampling_params),
                    use_tqdm=False
                )
        
        print("final full response", context+outputs[0].outputs[0].text) # TODO: delete 
        
        full_response = (context+outputs[0].outputs[0].text)[len(prompt):]
        return full_response
            
        
    def generate_response(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """process API request and generate response"""
        for message in body["messages"]:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                user_message = message["content"]

        if not system_prompt:
            raise ValueError("No system prompt found in the request.")
        if not user_message:
            raise ValueError("No user message found in the request.")
        
        assert body["model"] == self.model_config.model_path, f"model mismatch: {body['model']} != {self.model_config.model_path}"
        prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, 
                                                    {"role": "user", "content": user_message}],
                                                    add_generation_prompt=True,
                                                    tokenize=False)
        sampling_params = {
            "temperature": body["temperature"],
            "max_tokens": body["max_tokens"],
            "top_p": body["top_p"],
            "n": body["n"], 
            "stop": ["</s>", "<|im_end|>", "<|endoftext|>", "```output"], 
            "include_stop_str_in_output": True,
        }

        full_response = self.generate_with_tools(prompt, sampling_params)
        
        # format the response into OpenAI-compliant format
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
                        "content": f"{full_response}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  
                "completion_tokens": 0,
                "total_tokens": 0
            } 
        } # TODO: check benchmark format  