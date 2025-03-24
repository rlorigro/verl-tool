from .base import BaseTool, register_tool


@register_tool
class PythonCodeTool(BaseTool):
    tool_type = "python_code"
    
    def get_usage_inst(self):
        return "Usage instructions for PythonCodeTool"
    
    def parse_action(self, action:str):
        """
        Parse the raw action string (which is the llm response) into a actual action and it's contents
        """
        return action
    
    def conduct_action(self, trajectory_id, action, extra_field):
        parsed = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        observation = f"Base observation for {trajectory_id}"        
        return observation
    