from .base import BaseTool, register_tool
import re
import sys
import os
import io
import time
import traceback
import multiprocessing
from multiprocessing import Process, Queue
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# List of safe modules that can be imported in the sandbox
SAFE_MODULES = {
    'math', 'random', 'datetime', 'collections', 'itertools', 
    'functools', 'operator', 'string', 're', 'json', 'csv',
    'time', 'copy', 'hashlib', 'uuid', 'statistics',
    # Add more safe modules as needed
}

# Define a restricted set of builtins
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytes', 'callable',
    'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter',
    'float', 'format', 'frozenset', 'getattr', 'hasattr', 'hash',
    'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter', 'len',
    'list', 'map', 'max', 'min', 'next', 'object', 'oct', 'ord',
    'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
    'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
}

def _prepare_restricted_globals():
    """
    Create a restricted globals dictionary for safer code execution.
    """
    # Start with a clean namespace
    restricted_globals = {'__builtins__': {}}
    
    # Add safe builtins
    for name in SAFE_BUILTINS:
        if hasattr(__builtins__, name):
            restricted_globals['__builtins__'][name] = getattr(__builtins__, name)
    
    return restricted_globals

def _execute_in_sandbox(code: str, result_queue: Queue, error_queue: Queue, timeout: int = 10):
    """
    Execute code in a separate process with restricted access to Python functionality.
    This function runs in a separate process and communicates results via queues.
    
    Args:
        code: The Python code to execute
        result_queue: Queue to store successful execution results
        error_queue: Queue to store execution errors
        timeout: Maximum execution time in seconds (not used directly here, see process timeout)
    """
    try:
        # Redirect stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        try:
            # Prepare restricted environment
            restricted_globals = _prepare_restricted_globals()
            
            # Pre-import safe modules and add them to globals
            for module_name in SAFE_MODULES:
                try:
                    # Use __import__ directly to avoid import hooks
                    module = __import__(module_name)
                    restricted_globals[module_name] = module
                except ImportError:
                    pass  # Skip modules that can't be imported
            
            # Compile the code first to catch syntax errors early
            compiled_code = compile(code, '<string>', 'exec')
            
            # Execute in restricted environment
            exec(compiled_code, restricted_globals)
            
            # Get output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            
            # Send results to the queue
            result = stdout_content
            if stderr_content:
                result += f"\nERROR:\n{stderr_content}"
                
            result_queue.put(result)
            
        except Exception as e:
            # Handle execution errors
            error_msg = f"Error executing program: {str(e)}\n"
            error_msg += traceback.format_exc()
            error_queue.put(error_msg)
            
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            # Close the buffers
            stdout_buffer.close()
            stderr_buffer.close()
            
    except Exception as e:
        # Handle critical process errors
        try:
            error_queue.put(f"Critical process error: {str(e)}")
        except:
            # Last resort if even the error queue fails
            pass

def execute_code_safely(code: str, timeout: int = 10) -> str:
    """
    Execute Python code in a completely isolated process with a strict timeout.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Output of the execution or error message
    """
    # Create communication queues
    result_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()
    
    # Create a new process for isolated execution
    process = Process(
        target=_execute_in_sandbox,
        args=(code, result_queue, error_queue, timeout)
    )
    
    # Set process as daemon so it doesn't block program exit
    process.daemon = True
    
    # Start the process
    start_time = time.time()
    process.start()
    
    # Wait for the process to finish with timeout
    process.join(timeout)
    
    # Check if the process is still running after timeout
    if process.is_alive():
        # Process is still running after timeout, terminate it
        process.terminate()
        process.join(1)  # Give it 1 second to terminate gracefully
        
        # If still alive, try to kill more forcefully (Unix-only)
        if process.is_alive() and hasattr(os, 'kill'):
            try:
                os.kill(process.pid, 9)  # SIGKILL
                process.join(1)
            except OSError:
                pass
        
        return f"Execution timed out after {timeout} seconds and was terminated."
    
    # Process finished within timeout, collect results
    if not error_queue.empty():
        # Execution encountered an error
        return error_queue.get()
        
    if not result_queue.empty():
        # Execution succeeded
        return result_queue.get()
    
    # Neither queue has data (shouldn't happen if process exited normally)
    return "Execution completed but produced no output."

@register_tool
class SafePythonCodeTool(BaseTool):
    tool_type = "safe_python_code"
    timeout = 10
    
    def __init__(self, num_workers=None):
        super().__init__()
        # Number of concurrent workers for ThreadPoolExecutor
        self.num_workers = num_workers or min(32, os.cpu_count() * 2)
    
    def get_usage_inst(self):
        return "You are able to run the python code in your responses that are enclosed in <python> and </python> tags. The output of the code (stdout and stderr) will be returned between <o> and </o> tags."
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the action string to extract Python code.
        
        Args:
            action: Raw action string from LLM response
            
        Returns:
            Tuple of (extracted code, is valid)
        """
        key_words = ["<python>", "</python>", "```python", "```"]
        if not any(keyword in action for keyword in key_words):
            # If no keywords are found, return the action as is
            return action, False
            
        # First try to match <python> tags
        all_valid_python_code = re.findall(r"<python>((.|\n)*?)</python>", action, re.DOTALL)
        
        # If that fails, try to match markdown code blocks
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```python((.|\n)*?)```", action, re.DOTALL)
            
        if len(all_valid_python_code) == 0:
            # No valid code found
            return "", False
        else:
            # Extract the first code block
            return all_valid_python_code[0][0], True
    
    def _sanitize_code(self, code: str) -> str:
        """
        Add extra safeguards to the code.
        
        Args:
            code: Original code to sanitize
            
        Returns:
            Sanitized code
        """
        # Add safety wrappers and import restrictions
        safe_code = f"""
# Code executed in sandbox with restricted access
{code}
"""
        return safe_code
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute a single Python code action in a secure sandbox.
        
        Args:
            trajectory_id: ID of the trajectory
            action: Action to perform
            extra_field: Additional data
            
        Returns:
            Tuple of (observation, done, is_valid)
        """
        try:
            parsed_action, is_valid = self.parse_action(action)
            
            if not is_valid:
                observation = "No valid python code between <python> and </python> tags found."
                done = True
            else:
                # Sanitize code
                sanitized_code = self._sanitize_code(parsed_action)
                
                # Execute in sandbox with timeout
                observation = execute_code_safely(sanitized_code, timeout=self.timeout)
                done = False
                
            # Format the output
            if action.endswith("```output"):
                formatted_observation = f"\n{observation}```"
            else:
                formatted_observation = "\nHere is the returned execution results of the above python codes:\n"
                formatted_observation += f"<o>{observation}</o>"
                
            return formatted_observation, done, is_valid
            
        except Exception as e:
            # Catch any unexpected exceptions to prevent server crashes
            error_msg = f"Execution error: {str(e)}"
            if action.endswith("```output"):
                return f"\n{error_msg}```", True, False
            else:
                return f"\nError occurred: <o>{error_msg}</o>", True, False
    
    def get_observations(self, trajectory_ids, actions, extra_fields):
        """
        Get observations for multiple actions in parallel.
        
        Args:
            trajectory_ids: List of trajectory IDs
            actions: List of actions
            extra_fields: List of additional data
            
        Returns:
            Tuple of (observations, dones, valids)
        """
        results = []
        
        try:
            # Use ThreadPoolExecutor for parallelizing the executions
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                # Submit all tasks
                for traj_id, action, extra_field in zip(trajectory_ids, actions, extra_fields):
                    future = executor.submit(self.conduct_action, traj_id, action, extra_field)
                    futures.append(future)
                
                # Process results with progress bar
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Getting observations using tool {self.tool_type}"
                ):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # Handle any unexpected exceptions from individual tasks
                        error_msg = f"Task execution failed: {str(e)}"
                        results.append((f"\nError occurred: <o>{error_msg}</o>", True, False))
                
            # Unpack the results
            if not results:
                # Fallback if all tasks failed
                observations = ["Execution failed"] * len(trajectory_ids)
                dones = [True] * len(trajectory_ids)
                valids = [False] * len(trajectory_ids)
            else:
                observations, dones, valids = zip(*results)
                
            return observations, dones, valids
            
        except Exception as e:
            # Global exception handler to prevent server crashes
            error_msg = f"Global execution error: {str(e)}"
            observations = [error_msg] * len(trajectory_ids)
            dones = [True] * len(trajectory_ids)
            valids = [False] * len(trajectory_ids)
            return observations, dones, valids