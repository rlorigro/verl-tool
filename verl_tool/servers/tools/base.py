from pathlib import Path
registered_tools = {}
ALL_TOOLS = []

def get_tool_cls(tool_type):
    if tool_type in ALL_TOOLS:
        if tool_type == "base":
            return BaseTool
        __import__(f".{tool_type}", globals(), locals(), [tool_type], 0)
        tool_class = registered_tools.get(tool_type)
        return tool_class
    else:
        raise ValueError(f"Tool type {tool_type} not found. Available tools: {ALL_TOOLS}")


def register_tool(cls):
    """
    Decorator to register a tool class in the registered_tools dictionary.
    The class is registered using its tool_type attribute.
    """
    tool_type = getattr(cls, 'tool_type', cls.__name__)
    registered_tools[tool_type] = cls
    return cls

class BaseTool:
    tool_type = __name__
    
    def __init__(self, tool_type=None):
        self.tool_type = tool_type if tool_type else self.__class__.tool_type
        registered_tools[self.tool_type] = self.__class__
    
    def get_observations(self, trajectory_id, actions, queries, extra_data):
        return [f"Base observation for {trajectory_id}"]


# go through all files in the tools directory and register them
cur_dir = Path(__file__).parent
excluding_files = ["__init__.py", "base.py"]
ALL_TOOLS.append("base")
for file in cur_dir.iterdir():
    if file.is_file() and file.name not in excluding_files:
        ALL_TOOLS.append(file.stem)

# Print all registered tools
print("Registered Tools:")
for tool in ALL_TOOLS:
    print(f"  - {tool}")
    