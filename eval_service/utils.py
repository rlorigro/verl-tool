import re
import requests
import uuid
from typing import List, Dict, Tuple, Any, Optional

def extract_final_code(text: str) -> str:
    """从模型最终输出中提取代码块"""
    # 查找markdown代码块
    code_blocks = re.findall(r"```python\s*([\s\S]*?)```", text)
    
    if code_blocks:
        return code_blocks[-1].strip(), True  # 返回最后一个代码块
    
    else:
        return None, False
    
    

def extract_python_tags(text: str) -> Tuple[Optional[str], bool]:
    """从文本中提取<python></python>标签内容"""
    pattern = r"<python>((.|\n)*?)</python>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0][0], True
    else:
        return None, False
    
def call_tool_server(server_url: str, trajectory_id: str, python_code: str, finish: bool = False) -> Dict:
    """调用工具服务器执行Python代码"""
    # 格式化动作（用<python>标签包装代码）
    action = f"<python>{python_code}</python>"
    
    # 准备请求数据
    data = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "finish": [finish]
    }
    
    try:
        # 调用API
        response = requests.post(server_url, json=data)
        response.raise_for_status()
        
        # 解析返回结果
        result = response.json()
        observation = result["observations"][0]
        done = bool(result["dones"][0])
        valid = bool(result["valids"][0])
        
        return {
            "observation": observation,
            "done": done,
            "valid": valid
        }
    except Exception as e:
        return {
            "observation": f"Error calling tool server: {str(e)}",
            "done": True,
            "valid": False
        }