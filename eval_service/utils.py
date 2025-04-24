import re
import requests
import uuid
from typing import Dict, Tuple, Optional

def extract_python_tags(text: str) -> Tuple[Optional[str], bool]:
    code_blocks = re.findall(r"```python\s*([\s\S]*?)```", text)
    
    if code_blocks:
        return code_blocks[-1].strip(), True  # return the most recent (last) code block
    
    else:
        return None, False
    
def call_tool_server(server_url: str, trajectory_id: str, action: str) -> Dict:
    """querying the tool server for the observation and done flag"""
    # perpare payload
    data = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
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