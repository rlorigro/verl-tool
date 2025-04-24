import os
import json
import argparse
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from config import ServerConfig, ModelConfig, ToolConfig
from model_service import ModelService

# Define API request models
class ChatMessage(BaseModel):
    role: str  # Role of the message (system, user, assistant)
    content: str  # Content of the message

class ChatCompletionRequest(BaseModel):
    model: str  # Model identifier
    messages: List[ChatMessage]  # List of conversation messages
    temperature: Optional[float] = 0.7  # Controls randomness in generation
    top_p: Optional[float] = 0.9  # Nucleus sampling parameter
    max_tokens: Optional[int] = 1024  # Maximum number of tokens to generate

def create_app(config: ServerConfig):
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
    model_service = ModelService(config.llm_config, config.tool_config)
    
    # Store service in application state
    app.state.model_service = model_service
    
    @app.on_event("startup")
    async def startup_event():
        """Load model when application starts"""
        app.state.model_service.load_model()
        print(f"Model loaded: {config.llm_config.model_path}")
    
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        """
        Chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses
        with tool calling capabilities
        """
        try:
            request_body = await request.json()
            response = app.state.model_service.generate_response(request_body)
            
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify service availability"""
        return {"status": "healthy"}
    
    return app

def parse_args():
    parser = argparse.ArgumentParser(description='Verl-tool server configuration')
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                      help='Path to the model')
    parser.add_argument('--tool-server-url', type=str, default='http://localhost:30245/get_observation',
                      help='URL for tool server')
    parser.add_argument('--max-turns', type=int, default=5,
                      help='Maximum number of tool interaction turns')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to bind the server')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to bind the server')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration from command line arguments
    config = ServerConfig(
        llm_config=ModelConfig(
            model_path=args.model_path,  # Path to the model
        ),
        tool_config=ToolConfig(
            tool_server_url=args.tool_server_url,  # URL for tool server
            # valid_actions=json.loads(args.valid_actions),  # List of valid tool actions
            max_turns=args.max_turns,  # Maximum number of tool interaction turns
        ),
        host=args.host,  # Host to bind the server
        port=args.port,  # Port to bind the server
    )
    
    # Create and run the application
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()