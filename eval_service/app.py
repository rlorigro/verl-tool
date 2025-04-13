import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import json

from config import ServerConfig, ModelConfig, ToolConfig
from model_service import ModelService

# 定义API请求模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1024

def create_app(config: ServerConfig):
    """创建并配置FastAPI应用"""
    app = FastAPI(
        title="LLM代码工具服务",
        description="兼容OpenAI API的大模型代码工具调用服务",
        version="1.0.0"
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 初始化模型服务
    model_service = ModelService(config.llm_config, config.tool_config)
    
    # 存储服务到应用状态
    app.state.model_service = model_service
    
    @app.on_event("startup")
    async def startup_event():
        """启动时加载模型"""
        app.state.model_service.load_model()
        print(f"模型已加载: {config.llm_config.model_path}")
    
    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, req: Request):
        """兼容OpenAI的聊天完成API端点"""
        try:
            # 将Pydantic模型转换为字典
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # 生成响应
            response = app.state.model_service.generate_response(messages)
            
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "healthy"}
    
    return app

if __name__ == "__main__":
    # 从环境变量加载配置
    config = ServerConfig(
        llm_config=ModelConfig(
            model_path=os.environ.get("MODEL_PATH", "yi-34b-chat"),
        ),
        tool_config=ToolConfig(
            tool_server_url=os.environ.get("TOOL_SERVER_URL", "http://localhost:30286/get_observation"),
            valid_actions=json.loads(os.environ.get("VALID_ACTIONS", '["python"]')),
            max_turns=int(os.environ.get("MAX_TURNS", "5")),
        ),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
    
    # 创建并运行应用
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)