import os
import json
import argparse
import inspect
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from transformers import HfArgumentParser
import uvicorn

from config import ServerConfig, ModelConfig, ToolConfig
from model_service import ModelService

def create_app(server_config: ServerConfig, model_config: ModelConfig, tool_config: ToolConfig) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        config: Server configuration object
        
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="LLM Code Tool Service",
        description="Large language model code tool calling service compatible with OpenAI API",
        version="1.0.0"
    )
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize the model service
    model_service = ModelService(model_config, tool_config)
    model_service.load_model()
    
    # Store service in application state
    app.state.model_service = model_service
    
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        """
        Chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses
        with tool calling capabilities
        """
        try:
            request_body = await request.json()
            response = await app.state.model_service.generate_response_async(request_body)
            
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify service availability"""
        return {"status": "healthy"}
    
    return app

async def main_async():
    hf_parser = HfArgumentParser((ServerConfig, ModelConfig, ToolConfig))
    server_config, model_config, tool_config = hf_parser.parse_args_into_dataclasses()
    tool_config.post_init()
    # Create and run the application
    app = create_app(server_config, model_config, tool_config)
    
    config = uvicorn.Config(app, host=server_config.host, port=server_config.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

def main():
    import asyncio
    asyncio.run(main_async())

if __name__ == "__main__":
    main()