"""
Tool Server - A FastAPI server to manage and execute tools based on incoming requests.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Set, Union

import fire
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from .tools import get_tool_cls, ALL_TOOLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Data Models ----

class ExtraField(BaseModel):
    """A model for extra fields in requests"""
    finish: bool = False
    # Add other fields as needed

class AgentRequest(BaseModel):
    """Model for incoming agent requests"""
    trajectory_ids: List[Union[str, int]]  # Allow both string and integer IDs
    actions: List[str]
    extra_fields: Optional[List[Dict[str, Any]]] = None
    
    @validator('trajectory_ids', each_item=True)
    def convert_trajectory_ids_to_string(cls, v):
        # Convert any non-string IDs to strings
        return str(v)
    
    @validator('extra_fields', pre=True, always=True)
    def set_extra_fields(cls, v, values):
        if v is None and 'actions' in values:
            return [{}] * len(values['actions'])
        return v
    
    @validator('extra_fields')
    def validate_length(cls, v, values):
        if 'actions' in values and len(v) != len(values['actions']):
            raise ValueError(f"Number of extra_fields ({len(v)}) does not match number of actions ({len(values['actions'])})")
        return v

class AgentResponse(BaseModel):
    """Model for outgoing agent responses"""
    observations: List[str]
    dones: List[bool]
    valids: List[bool]


# ---- Tool Management ----

class ToolManager:
    """Manages all tools and their execution"""
    
    def __init__(self, tool_types: Tuple[str], num_workers_per_tool: int = 4):
        """
        Initialize the tool manager with specified tools
        
        Args:
            tool_types: Tuple of tool type names to initialize
            num_workers_per_tool: Number of workers for each tool
        """
        self.tools: Dict[str, Any] = {}
        self._initialize_tools(tool_types, num_workers_per_tool)
        
    def _initialize_tools(self, tool_types: Tuple[str], num_workers: int) -> None:
        """Initialize all tools based on tool types"""
        # Ensure we have the finish tool
        if "finish" not in tool_types:
            tool_types = tool_types + ("finish",)
            
        logger.info(f"Initializing tools: {tool_types}")
        for tool_type in tool_types:
            try:
                tool_cls = get_tool_cls(tool_type)
                self.tools[tool_type] = tool_cls(num_workers=num_workers)
                logger.info(f"Initialized tool: {tool_type}")
            except Exception as e:
                logger.error(f"Failed to initialize tool {tool_type}: {e}")
                
        # Log available vs. active tools
        logger.info("Available Tools:")
        for tool in ALL_TOOLS:
            status = "active" if tool in self.tools else "inactive"
            logger.info(f"  - {tool}: {status}")
    
    def get_tool_usage_instructions(self) -> str:
        """Get usage instructions for all available tools"""
        usage_instructions = {}
        for tool_type, tool in self.tools.items():
            if tool_type not in ["finish", "base"]:
                usage_instructions[tool_type] = tool.get_usage_inst()
                
        message = "Your action did not match any of the available tools, please use one of the following tools: \n"
        message += "\n".join([f"- {tool_type}: {usage_instructions[tool_type]}" for tool_type in usage_instructions])
        return message
    
    def identify_tool_for_action(self, action: str, extra_field: Dict[str, Any]) -> Optional[str]:
        """
        Identify which tool should process a given action
        
        Args:
            action: The action string to process
            extra_field: Extra fields associated with the action
            
        Returns:
            The identified tool type or None if no tool matches
        """
        return "python_code"
        # Check for finish condition
        if extra_field.get("finish", False):
            return "finish"
            
        # If only one tool available, use it
        if len(self.tools) == 1:
            return list(self.tools.keys())[0]
            
        # Try to find matching tool
        for tool_type, tool in self.tools.items():
            _, valid = tool.parse_action(action)
            if valid:
                return tool_type
                
        return None
    
    def process_actions(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        Process a batch of actions using appropriate tools
        
        Args:
            trajectory_ids: List of trajectory IDs
            actions: List of action strings
            extra_fields: List of extra fields for each action
            
        Returns:
            Tuple of (observations, dones, valids) lists
        """
        # Identify which tool should process each action
        tool_types = []
        from tqdm import tqdm
        for i in tqdm(range(len(actions)), desc="Identifying tool types"):
            tool_type = self.identify_tool_for_action(actions[i], extra_fields[i])
            tool_types.append(tool_type)
        logger.debug(f"Identified tool types: {tool_types}")
        
        # Prepare result containers
        all_observations = [None] * len(actions)
        all_dones = [False] * len(actions)
        all_valids = [False] * len(actions)
        
        # Group actions by tool type for batch processing
        unique_tool_types: Set[Optional[str]] = set(tool_types)
        
        for tool_type in unique_tool_types:
            # Get indices of actions for this tool type
            indices = [i for i, t in enumerate(tool_types) if t == tool_type]
            
            if tool_type is None:
                # Handle actions that don't match any tool
                usage_instructions = self.get_tool_usage_instructions()
                observations = [usage_instructions] * len(indices)
                dones = [False] * len(indices)
                valids = [False] * len(indices)
            else:
                # Process with the appropriate tool
                tool = self.tools[tool_type]
                tool_trajectory_ids = [trajectory_ids[i] for i in indices]
                tool_actions = [actions[i] for i in indices]
                tool_extra_fields = [extra_fields[i] for i in indices]
                
                observations, dones, valids = tool.get_observations(
                    tool_trajectory_ids, tool_actions, tool_extra_fields
                )
            
            # Store results in the appropriate positions
            for idx_pos, result_idx in enumerate(indices):
                all_observations[result_idx] = observations[idx_pos]
                all_dones[result_idx] = dones[idx_pos]
                all_valids[result_idx] = valids[idx_pos]
                
        return all_observations, all_dones, all_valids


# ---- Server Implementation ----

class ToolServer:
    """Server to handle tool execution requests"""
    
    def __init__(
        self,
        tool_types: Tuple[str],
        host: str = "0.0.0.0",
        port: int = 5000,
        workers_per_tool: int = 4,
        max_concurrent_requests: int = 50,
    ):
        """
        Initialize the tool server
        
        Args:
            tool_types: Tool types to initialize
            host: Server host
            port: Server port
            workers_per_tool: Number of workers per tool
            max_concurrent_requests: Maximum number of concurrent requests
        """
        self.host = host
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize tool manager
        self.tool_manager = ToolManager(tool_types, workers_per_tool)
        
        # Initialize thread pool for concurrent processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Tool Server",
            description="A server for executing tools based on agent requests",
            version="1.0.0",
        )
        
        # Set up routes and event handlers
        self._configure_app()
        
    def _configure_app(self):
        """Configure FastAPI app with routes and event handlers"""
        
        # Create a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Request handling route
        @self.app.post("/get_observation", response_model=AgentResponse)
        async def handle_observation_request(request: Request, background_tasks: BackgroundTasks):
            async with semaphore:
                # Parse request
                data = await request.json()
                logger.debug(f"Received request: {data}")
                
                try:
                    # Handle raw request data first for more flexible input handling
                    # Convert any numeric trajectory_ids to strings
                    if "trajectory_ids" in data:
                        data["trajectory_ids"] = [str(tid) if not isinstance(tid, str) else tid 
                                                 for tid in data.get("trajectory_ids", [])]
                    
                    # Validate request with pydantic model
                    agent_request = AgentRequest.parse_obj(data)
                    
                    # Process the request asynchronously
                    observations, dones, valids = await self._process_request(
                        agent_request.trajectory_ids,
                        agent_request.actions,
                        agent_request.extra_fields
                    )
                    
                    # Create response
                    response = AgentResponse(
                        observations=observations,
                        dones=dones,
                        valids=valids
                    )
                    logger.debug(f"Sending response: {response}")
                    return response
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}", exc_info=True)
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"Failed to process request: {str(e)}"}
                    )
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
            
        # Shutdown event handler
        @self.app.on_event("shutdown")
        async def shutdown_event():
            logger.info("Shutting down server...")
            self.thread_pool.shutdown(wait=True)
            logger.info("Server shutdown complete")
    
    async def _process_request(
        self,
        trajectory_ids: List[str],
        actions: List[str],
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        Process a batch of actions asynchronously
        
        Args:
            trajectory_ids: List of trajectory IDs
            actions: List of action strings
            extra_fields: List of extra fields for each action
            
        Returns:
            Tuple of (observations, dones, valids) lists
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.tool_manager.process_actions,
            trajectory_ids,
            actions,
            extra_fields
        )
    
    def start(self):
        """Start the server"""
        logger.info(f"Starting server on {self.host}:{self.port}")
        logger.info(f"Server configured for up to {self.max_concurrent_requests} concurrent requests")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# ---- CLI Entry Point ----

def main(
    host: str = "0.0.0.0",
    port: int = 5000,
    workers_per_tool: int = 4,
    max_concurrent_requests: int = 50,
    tool_type: Union[str, Tuple[str]] = "base",
    log_level: str = "info",
):
    """
    Start the tool server
    
    Args:
        host: The host address
        port: The port number
        workers_per_tool: Number of workers per tool
        max_concurrent_requests: Maximum number of concurrent requests
        tool_type: Tool type(s) to use (comma-separated string or tuple)
        log_level: Logging level (debug, info, warning, error)
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.basicConfig(level=numeric_level)
    
    # Convert string to tuple of tool types if needed
    if isinstance(tool_type, str):
        if "," in tool_type:
            tool_type = tuple(t.strip() for t in tool_type.split(","))
        else:
            tool_type = (tool_type,)
    
    # Create and start server
    server = ToolServer(
        tool_types=tool_type,
        host=host,
        port=port,
        workers_per_tool=workers_per_tool,
        max_concurrent_requests=max_concurrent_requests,
    )
    server.start()


if __name__ == "__main__":
    fire.Fire(main)