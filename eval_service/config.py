from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

class ModelConfig(BaseModel):
    model_path: str
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1

class ToolConfig(BaseModel):
    tool_server_url: str = "http://localhost:30286/get_observation"
    valid_actions: List[str] = ["python"]  # 工具名称列表，用于检测结束标记
    max_turns: int = 1  # 最大工具调用轮次
    no_action_as_stop: bool = True  # 无工具调用视为停止
    min_action_num: int = 1  # 最少工具调用次数
    truncate_obs_side: str = "left"  # 截断方向
    max_prompt_length: int = 4096  # 最大提示长度
    max_obs_length: int = 1024  # 最大观察结果长度
    max_start_length: int = 4096  # 最大开始长度
    max_response_length: int = 4096  # 最大响应长度

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    llm_config: ModelConfig
    tool_config: ToolConfig
    execution_prompt: str = """\
Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside <python> and </python> tags. The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between <python> and </python> tags are independent program. You can run Python code as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\\nyour code here\\n```. The final program will be evaluated against the test cases.
"""