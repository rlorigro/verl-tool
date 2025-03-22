import argparse
import re
import fire

import torch
import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .tools import get_tool_cls

logger = logging.getLogger(__name__)


def parse_agent_request(data):
    trajectory_ids = data.get("trajectory_ids")
    actions = data.get("actions")
    queries = data.get("queries")
    extra_data = data.get("extra_data") 
    
    parsed_agent_request = {
        "trajectory_id": trajectory_ids,
        "actions": actions,
        "queries": queries,
        "extra_data": extra_data,
    }
    return parsed_agent_request


def main(
    host: str = "0.0.0.0",
    port: int = 5000,
    num_workers: int = 1,
    max_obs_length: int = 2048,
    tool_type = "base",
):

    app = FastAPI()
    tool = get_tool_cls(tool_type)()
    
    @app.post("/get_observation")
    async def get_observation(request: Request):
        data = await request.json()
        parsed_data = parse_agent_request(data)
        observations = tool.get_observations(**parsed_data)
        result = {"observations": observations}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)
    
    uvicorn.run(app, host=host, port=port, log_level="info")
    

if __name__ == "__main__":
    fire.Fire(main)