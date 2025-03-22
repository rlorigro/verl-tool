from .base import BaseTool, register_tool


@register_tool
class PythonCodeTool(BaseTool):
    tool_type = "python_code"
    
    def get_observations(self, trajectory_id, actions, queries, extra_data):
        return [f"Python code tool observation for {trajectory_id}"]
    
    def get_actions(self, trajectory_id, observations, queries, extra_data):
        return [f"Python code tool actions for {trajectory_id}"]
    