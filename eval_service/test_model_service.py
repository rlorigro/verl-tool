from model_service import ModelService
from config import ModelConfig, ToolConfig
import os
import json

llm_config=ModelConfig(
    model_path=os.environ.get("MODEL_PATH", "yi-34b-chat"),
)
tool_config=ToolConfig(
    tool_server_url=os.environ.get("TOOL_SERVER_URL", "http://localhost:30150/get_observation"),
    valid_actions=json.loads(os.environ.get("VALID_ACTIONS", '["output"]')),
    max_turns=int(os.environ.get("MAX_TURNS", "5")),
)

model_service = ModelService(llm_config, tool_config)

res = model_service.generate_with_tools(
    prompt = "This is the dummy prompt:\n",
    debug=True
)

print(res)