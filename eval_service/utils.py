import re
import requests
import uuid
from typing import List, Dict, Tuple, Any, Optional

def extract_python_tags(text: str) -> Tuple[Optional[str], bool]:
    code_blocks = re.findall(r"```python\s*([\s\S]*?)```", text)
    
    if code_blocks:
        return code_blocks[-1].strip(), True  # return the most recent (last) code block
    
    else:
        return None, False
    
def call_tool_server(server_url: str, trajectory_id: str, python_code: str, finish: bool = False) -> Dict:
    """querying the tool server for the observation and done flag"""
    # reformat the action: wrap the code with <python> and </python>
    action = f"<python>{python_code}</python>"
    
    # perpare payload
    data = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}]
    }
    
    try:
        response = requests.post(server_url, json=data)
        response.raise_for_status()
        
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
        print(f"Error calling tool server: {str(e)}")
        return {
            "observation": f"Error calling tool server: {str(e)}",
            "done": True,
            "valid": False
        }