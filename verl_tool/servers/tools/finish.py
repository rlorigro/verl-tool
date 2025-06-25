from .base import BaseTool, register_tool
import regex as re



@register_tool
class FinishTool(BaseTool):
    tool_type = "finish"
    timeout = 10
    
    def __init__(self, num_workers=1, other_tools:list = []):
        super().__init__(num_workers)
        self.other_tools = other_tools
    
    def get_usage_inst(self):
        return ""
    
    def parse_action(self, action:str):
        """
        Parse the raw action string to check for answer tags or finish conditions.
        """
        # Check for <answer> tags (Search-R1 style)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", action, re.DOTALL)
        if len(answer_matches) > 0:
            final_answer = answer_matches[-1].strip()
            return final_answer, True
        
        # Default behavior - trajectory ends without explicit answer
        return "", False
    
    def conduct_action(self, trajectory_id, action, extra_data):
        final_answer, is_valid = self.parse_action(action)
        
        # For Search-R1 style, we don't add observation when finishing with <answer>
        # The final answer is already in the LLM response
        observation = ""
        done = True
        
        # Clean up environments for all tools
        for tool in self.other_tools:
            if tool.has_env(trajectory_id):
                tool.delete_env(trajectory_id)
        
        return observation, done, is_valid
    