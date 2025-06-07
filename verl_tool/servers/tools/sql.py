from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import signal
import sys
import json
import uuid
import hashlib
from typing import Tuple, Dict, Any, Optional
from ..utils import kill_python_subprocess_processes
from .sql_executor import score, Executor
import random

# Timeout for code execution in seconds
TIMEOUT = 5

import concurrent.futures

def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    if kwargs is None:
        kwargs = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise Exception(f"Function timed out after {timeout} seconds")

@register_tool
class SqlTool(BaseTool):
    tool_type = "sql"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = False
    enable_mannual_reflection = False
    force_run_test_cases = False
    done_without_error = False
    executor = Executor()
    def get_usage_inst(self):
        return "You are able to write and execute Python code."
    
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
        # print('=====> sql ')
        code = re.findall(r"(```sql.*?```)", action, re.DOTALL)
        
        if len(code) == 0:
            # code = [action]
            return "Error", False
        
        parsed_code = code[-1].strip()

        return parsed_code, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        
        parsed_action, is_valid = self.parse_action(action)# self.parse_action(self, action)
        
        env = self.load_env(trajectory_id)
        
        
        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = "Code Extraction Error: code block not detected."
            execution_result = ""
            done = False
            valid = False
            correctness = 0.0
        else:
            # Extract stdin if provided in extra_field
            gold = extra_field.get("gt_sql", None) if extra_field else None
            db_id = extra_field.get("db_id", None)
            code_to_execute = parsed_action
            
            meta = {
            "db_id": db_id, #"thrombosis_prediction/thrombosis_prediction.sqlite", 
            "gold_sql": gold, # "SELECT T2.`T-BIL`, T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID ORDER BY T2.`T-IL` DESC LIMIT 1", 
            "cmp_method": "bird"}
            try:
                # correctness, error_message = run_with_timeout(score, args=(code_to_execute, meta), timeout=2)
                correctness, error_message = score(code_to_execute, meta, self.executor)
            except Exception as e:
                correctness = 0.0
                error_message = "Execution Timeout."

            if error_message:
                observation = f"```error\n{error_message}\n```\n"
                has_error = True
            else:
                observation = f""
                has_error = False
            # import pdb; pdb.set_trace()
            

            if self.done_without_error:
                if has_error:
                    done = False
                else:
                    done = True
            else: 
                done = False
            valid = True
        execution_result = observation # json.dumps({'correctness':correctness, 'message':observation})
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        # print(f"===> parsed code", parsed_action)
        print(f"===> resulting observation", execution_result)
        # return observation, done, valid
        return {'extracted':code_to_execute, 'correctness':correctness, 'message':observation}, done, valid
        