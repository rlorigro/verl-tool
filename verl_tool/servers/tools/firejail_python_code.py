# Firejail is a local sandbox and strikes the best balance of reliability, scalability, and security
# https://github.com/netblue30/firejail
# sudo add-apt-repository ppa:deki/firejail
# sudo apt-get update
# sudo apt-get install firejail firejail-profiles
from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import timeout_decorator
import sys
from typing import Tuple, Dict, Any, Optional
from ..utils import kill_python_subprocess_processes

# Timeout for code execution in seconds
TIMEOUT = 10

def check_forbidden_imports(code: str) -> bool:
    """
    Checks if the code contains imports of potentially dangerous packages.
    
    Args:
        code: Python code string to analyze
        
    Returns:
        Boolean indicating if the code contains forbidden imports
    """
    # List of potentially dangerous modules that could affect the host system
    forbidden_modules = [
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    ]
    
    # Simple string-based check for import statements
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True
    
    # Check for os.system, os.popen, and similar dangerous calls
    dangerous_patterns = [
        "os.system", "os.popen", "os.spawn", "os.fork", 
        "os.exec", "sys.exit", "os._exit", "os.kill"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    
    return False

@timeout_decorator.timeout(TIMEOUT, use_signals=False)
def _exec_firejail_with_timeout(code: str, stdin: Optional[str] = None) -> str:
    """
    Execute Python code in a Firejail sandbox with a timeout.
    
    Args:
        code: Python code string to execute
        stdin: Optional input to provide to the executed code
        
    Returns:
        String containing execution output
    """
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]  # avoid importing wrong stuff
    
    # Build the firejail command with resource limits and cleanup options
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",  # Limit file size
        "--rlimit-as=4096m",
    ]
    command.extend(["python3", "-c", code])
    
    result = subprocess.run(
        command,
        input=stdin.encode() if stdin else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False
    )
    
    stdout = result.stdout.decode()
    stderr = result.stderr.decode().strip()
    
    if result.returncode == 0:
        return stdout
    
    return f"{stdout}\nERROR:\n{stderr}"

def execute_python_in_firejail(code: str, stdin: Optional[str] = None) -> str:
    """
    Wrapper function to execute Python code in Firejail with exception handling.
    
    Args:
        code: Python code string to execute
        stdin: Optional input to provide to the executed code
        
    Returns:
        String containing execution output or error message
    """
    try:
        return _exec_firejail_with_timeout(code, stdin)
    except timeout_decorator.TimeoutError:
        return "Execution timed out after {} seconds.".format(TIMEOUT)
    except Exception as e:
        return f"Exception during execution: {str(e)}"


@register_tool
class FirejailPythonCodeTool(BaseTool):
    tool_type = "firejail_python_code"
    timeout = TIMEOUT
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code securely inside a Firejail sandbox."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        # Try to find Python code in various formats
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```python(.*?)```", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```(.*?)```", action, re.DOTALL)
        
        if len(all_valid_python_code) == 0:
            return "", False
        
        # Use the first code block found (we could extend this to support multiple blocks)
        parsed_code = all_valid_python_code[0].strip()
        
        return parsed_code, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action in a Firejail sandbox.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        
        if not is_valid:
            observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            return observation, True, False
        
        # Check for forbidden imports first
        if check_forbidden_imports(parsed_action):
            observation = "Execution blocked: Code contains potentially dangerous operations or imports."
            return observation, True, False
        
        # Extract stdin if provided in extra_field
        stdin = extra_field.get("stdin", None) if extra_field else None
        
        # Execute the code using firejail
        try:
            execution_result = execute_python_in_firejail(parsed_action, stdin)
            
            # Format the result
            if "ERROR:" in execution_result:
                observation = f"Execution completed with errors:\n{execution_result}"
            else:
                observation = f"Execution result:\n{execution_result}"
                
            return observation, False, True
            
        except Exception as e:
            observation = f"Error during execution: {str(e)}"
            return observation, True, False
        
    def get_observations(self, trajectory_ids, actions, extra_fields):
        # Get results from the parent class implementation
        results = super().get_observations(trajectory_ids, actions, extra_fields)
        
        # Kill any lingering Python processes
        killed_count = kill_python_subprocess_processes()
        if killed_count > 0:
            print(f"Terminated {killed_count} lingering Python execution processes")
        
        return results