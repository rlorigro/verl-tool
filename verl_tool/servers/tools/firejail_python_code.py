from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import signal
import sys
from typing import Tuple, Dict, Any, Optional
from ..utils import kill_python_subprocess_processes

import random

# Timeout for code execution in seconds
TIMEOUT = 5

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
    
def execute_python_in_firejail(code: str, timeout: int=TIMEOUT, stdin: Optional[str] = None) -> str:
    """
    Execute Python code in a Firejail sandbox with a timeout.
    
    Args:
        code: Python code string to execute
        stdin: Optional input to provide to the executed code
        
    Returns:
        String containing execution output or error message
    """
    # Check for forbidden imports first
    if check_forbidden_imports(code):
        return "Execution blocked: Code contains potentially dangerous operations or imports."
    
    # Create a minimal environment instead of copying everything
    original_env = os.environ.copy()
    env = {}
    
    # Core system variables
    essential_vars = [
        "PATH", "HOME", "USER", "SHELL", 
        "LANG", "LC_ALL", "LC_CTYPE", "TERM",
        # Python-specific
        "PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
        # Runtime optimization
        "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        # Temp directories
        "TMPDIR", "TEMP", "TMP",
        # Display if needed
        "DISPLAY", "XAUTHORITY"
    ]
    
    # Copy only essential variables if they exist
    for var in essential_vars:
        if var in original_env:
            env[var] = original_env[var]
    
    # Explicitly set optimization variables
    env["OPENBLAS_NUM_THREADS"] = "1"
    
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]
    
    # Build the firejail command with resource limits
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",  # Limit file size
        "--rlimit-as=1096m",
    ]
    command.extend(["python3", "-c", code])
    
    try:
        result = subprocess.run(
            command,
            input=stdin if stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            timeout=timeout
        )
        
        stdout = result.stdout
        stderr = result.stderr.strip()
        
        result = f"{stdout}\nError:\n{stderr}" if stderr else stdout
        if result:
            result = result.strip()
    except subprocess.TimeoutExpired:
        result = f"Execution timed out after {timeout} seconds.\n"
    return result

@register_tool
class FirejailPythonCodeTool(BaseTool):
    tool_type = "firejail_python_code"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>"]
    enable_history_code_execution = True
    enable_mannual_reflection = True
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
        env = self.load_env(trajectory_id)
        
        heuristic_sentences = {
            "empty": [
                "Hmm, no output at all. Since nothing broke, I'll draft a few edge-case inputs to see if the function ever emits data:",
                "The run is silent—no errors, no text. I'll broaden the test suite and watch for any change in behaviour:",
                "Blank output suggests the happy path passed; I'll now probe unusual parameters to confirm:",
                "Nothing was printed, yet the call completed. Let me invent some stress tests to check hidden branches:",
                "Zero output but no crash: time to craft randomized cases and observe whether output appears under different conditions:"
            ], # normally when there is just a function of code, that's the coding question most of the time
            "timeout": [
                "The call never returned—likely stuck in a heavy loop. I'll scan the control flow and think about where it could stall.",
                "Timeout reached. That hints at an expensive section or possible infinite recursion; I'll trace the algorithm paths and rethink them.",
                "Execution exceeded the limit. I'll review the data size assumptions and consider simpler test inputs first.",
                "Ran out of time—maybe I'm missing a termination condition. I'll inspect loops and add safeguards before retrying.",
                "Process froze long enough to trigger a timeout; I'll look for bottlenecks and refactor the slow part."
            ],
            "error": [
                "The code seems crashed. I'll read the stack trace, locate the failing line, and reason out a fix. Based on the error, I think",
                "Wait, I got some errors of my previous code, I'll double-check code logic and try to fix the root cause. Based on the error, I think",
                "Oops, the code crashed. I'll analyze the error message and see if I can fix it. Based on the error, I think",
            ],
            "success": [
                "Code executed successfully! I'll cross-check it with expectations and decide if more cases are needed. Based on the output, I think",
                "Now I have the execution result. After checking the output, I think",
                "Good, the code run successfully without error. However, are there any more corner cases that didn't cover?",
                "It looks like the code is working! But does this match my expected output?",
            ]
        }
        
        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            
            # Extract stdin if provided in extra_field
            stdin = extra_field.get("stdin", None) if extra_field else None
            
            if self.enable_history_code_execution:
                # Execute the code
                previous_parsed_code = [obs["action"] for obs in env["previous_obs"] if obs["is_valid"] and "error:" not in obs["observation"].lower()]
                code_to_execute = "\n".join(previous_parsed_code) + "\n" + parsed_action
                execution_result = execute_python_in_firejail(code_to_execute, self.timeout, stdin)
                # print("------")
                # print(code_to_execute)
                # print("------")
                # print("------")
                # print(execution_result)
                # print("------")
                # print("----")
                # print([obs["observation"] for obs in env["previous_obs"]])
                # print("----")
                for previous_obs in env["previous_obs"]:
                    if previous_obs["is_valid"] and "error:" not in previous_obs["observation"].lower():
                        execution_result = execution_result.replace(previous_obs["observation"], "", 1)
            else:
                execution_result = execute_python_in_firejail(parsed_action, self.timeout, stdin)
                
            execution_result = execution_result.lstrip(' \n')
            
            # Format the result
            if "Execution timed out" in execution_result:
                observation = execution_result
            else:
                observation = f"{execution_result}"
            done = False
            valid = True
        
        if action.endswith("```output"):
            observation = observation + "\n```"
        if action.endswith("<output>"):
            observation = observation + "\n</output>"
        if action.endswith("</python>"):
            observation = "<output>" + observation + "\n</output>"
        if action.strip(' \n').endswith("```"):
            observation = "```output\n" + observation + "\n```"
        
        observation = "\n" + observation + "\n"
        
        if valid and self.enable_mannual_reflection:
            # case: empty (correctly runned or the test case does not have output, need to check)
            if execution_result == "":
                # randomly select a sentence from the empty heuristic sentences
                idx = random.randint(0, len(heuristic_sentences["empty"]) - 1)
                observation += heuristic_sentences["empty"][idx]
            # case: execution timed out, need to check if the code is correct
            elif "Execution timed out" in execution_result:
                # observation = execution_result
                idx = random.randint(0, len(heuristic_sentences["timeout"]) - 1)
                observation += heuristic_sentences["timeout"][idx]
            # case: execution ends with error, need to look back and fix the bug
            elif "error:" in execution_result.lower():
                # observation = f"Execution completed with errors:\n{execution_result}"
                idx = random.randint(0, len(heuristic_sentences["error"]) - 1)
                observation += heuristic_sentences["error"][idx]     
            # case: generated output without error, need to check the code's output
            else:
                # observation = f"Execution result:\n{execution_result}"
                idx = random.randint(0, len(heuristic_sentences["success"]) - 1)
                observation += heuristic_sentences["success"][idx]
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
        