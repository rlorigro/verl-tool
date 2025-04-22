import ray
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from fastapi import FastAPI, Request
import uvicorn
from collections import defaultdict

# Initialize Ray
if not ray.is_initialized():
    try:
        ray.init(ignore_reinit_error=True)
    except:
        # Connect to existing Ray cluster
        ray.init(address="auto", ignore_reinit_error=True)

# Import your tool classes
from .tools import get_tool_cls, ALL_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ray actors for tools
@ray.remote
class RayToolWorker:
    """Ray actor for executing tool operations"""
    
    def __init__(self, tool_type: str):
        """
        Initialize a worker for a specific tool type.
        
        Args:
            tool_type: The type of tool this worker will handle
        """
        from .tools import get_tool_cls
        self.tool_type = tool_type
        # Create a single-worker tool instance (parallelization handled by Ray)
        self.tool = get_tool_cls(tool_type)()
        
    def execute(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
        """
        Execute a single tool action.
        
        Args:
            trajectory_id: Unique identifier for the trajectory
            action: The action string to execute
            extra_field: Additional data for the action
            
        Returns:
            tuple: (observation, done, valid) result of the action
        """
        return self.tool.conduct_action(trajectory_id, action, extra_field)
        
    def parse_action(self, action: str):
        """
        Check if this tool can handle the action.
        
        Args:
            action: The action string to parse
            
        Returns:
            tuple: (parsed_action, valid)
        """
        return self.tool.parse_action(action)
    
    def get_usage_inst(self):
        """
        Get usage instructions for this tool.
        
        Returns:
            str: The usage instructions
        """
        return self.tool.get_usage_inst()


class RayToolManager:
    """Tool manager that uses Ray for distributed execution"""
    
    def __init__(self, tool_types: Tuple[str], num_workers_per_tool: int = 4):
        """
        Initialize tool workers as Ray actors.
        
        Args:
            tool_types: Types of tools to initialize
            num_workers_per_tool: Number of Ray workers to create per tool type
        """
        self.tool_types = tool_types
        self.workers_per_tool = num_workers_per_tool
        self.tool_workers = {}
        
        # Make sure Ray is initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        self._initialize_tools()
        
    def _initialize_tools(self):
        """Initialize Ray actors for each tool type"""
        for tool_type in self.tool_types:
            if tool_type == "finish":
                continue  # Handle finish tool separately
                
            # Create multiple workers for each tool type for parallelization
            self.tool_workers[tool_type] = [
                RayToolWorker.remote(tool_type) 
                for _ in range(self.workers_per_tool)
            ]
        
        # Initialize finish tool (if needed)
        if "finish" not in self.tool_workers:
            self.tool_workers["finish"] = [
                RayToolWorker.remote("finish") 
                for _ in range(self.workers_per_tool)
            ]
            
        # Log available vs. active tools with emoji indicators
        logger.info("Available Tools:")
        for tool in ALL_TOOLS:
            if tool in self.tool_workers:
                status = "active 🟢"  # Green circle for active tools
                logger.info(f"  - {tool}: {status}")
            else:
                status = "inactive ⚪"  # White circle for inactive tools
                logger.info(f"  - {tool}: {status}")
    
    async def identify_tool_for_action(self, action: str, extra_field: Dict[str, Any]) -> Optional[str]:
        """
        Identify which tool type can handle this action.
        
        Args:
            action: The action string
            extra_field: Additional data for the action
            
        Returns:
            str or None: The identified tool type, or None if no tool matches
        """
        # Check for finish condition
        if extra_field.get("finish", False):
            return "finish"
            
        # Try each tool type with Ray
        for tool_type, workers in self.tool_workers.items():
            if tool_type == "finish":
                continue
                
            # Use the first worker to check validity
            _, valid_ref = await asyncio.to_thread(
                ray.get, workers[0].parse_action.remote(action)
            )
            
            if valid_ref:
                return tool_type
                
        return None
        
    def get_tool_usage_instructions(self) -> str:
        """
        Get usage instructions for all tools.
        
        Returns:
            str: Combined usage instructions for all tools
        """
        # Collect usage instructions from one worker of each type
        futures = {
            tool_type: workers[0].get_usage_inst.remote()
            for tool_type, workers in self.tool_workers.items()
            if tool_type != "finish"
        }
        
        instructions = ray.get(list(futures.values()))
        tool_types = list(futures.keys())
        
        message = "\nYour action did not match any of the available tools, please use one of the following tools: \n"
        message += "\n".join([f"- {tool_types[i]}: {instructions[i]}" for i in range(len(tool_types))])
        return message
        
    async def process_actions(
        self, 
        trajectory_ids: List[str], 
        actions: List[str], 
        extra_fields: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        Process actions using Ray workers in parallel.
        
        Args:
            trajectory_ids: List of trajectory IDs
            actions: List of actions corresponding to each trajectory
            extra_fields: List of extra data for each action
            
        Returns:
            tuple: (observations, dones, valids) lists for all actions
        """
        # Identify tool types for actions
        tool_types = []
        for i in range(len(actions)):
            tool_type = await self.identify_tool_for_action(actions[i], extra_fields[i])
            tool_types.append(tool_type)
            
        # Prepare results
        observations = [None] * len(actions)
        dones = [False] * len(actions)
        valids = [False] * len(actions)
        
        @ray.remote
        def non_tool_action(trajectory_id: str, action: str, extra_field: Dict[str, Any]):
            return self.get_tool_usage_instructions(), False, False
        
        pending_refs = []
        tool_worker_idx = defaultdict(int)
        for i, tool_type in enumerate(tool_types):
            trajectory_id = trajectory_ids[i]
            action = actions[i]
            extra_field = extra_fields[i]
            
            if tool_type is None:
                # Handle actions with no matching tool
                result_ref = non_tool_action.remote(trajectory_id, action, extra_field)
            else:
                workers = self.tool_workers[tool_type]
                worker_idx = tool_worker_idx[tool_type]
                tool_worker_idx[tool_type] = (worker_idx + 1) % len(workers)
                worker = workers[worker_idx]
                result_ref = worker.execute.remote(trajectory_id, action, extra_field)
            pending_refs.append(result_ref)
        
        # Get results as they complete
        if pending_refs:
            # Use asyncio to wait for Ray tasks
            results = [ray.get(ref) for ref in pending_refs]
            for i, result in enumerate(results):
                observation, done, valid = result
                observations[i] = observation
                dones[i] = done
                valids[i] = valid
        assert observations.count(None) == 0, f"{observations.count(None)} actions did not return an observation"
        return observations, dones, valids


# FastAPI server
class RayToolServer:
    """FastAPI server that uses Ray for distributed tool execution"""
    
    def __init__(
        self, 
        tool_types: Tuple[str], 
        host: str = "0.0.0.0", 
        port: int = 5000,
        workers_per_tool: int = 4,
        max_concurrent_requests: int = 64
    ):
        """
        Initialize the server.
        
        Args:
            tool_types: Types of tools to initialize
            host: Host address
            port: Port number
            workers_per_tool: Number of Ray workers per tool type
            max_concurrent_requests: Maximum number of concurrent requests
        """
        self.host = host
        self.port = port
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize Ray tool manager
        self.tool_manager = RayToolManager(tool_types, workers_per_tool)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Ray Tool Server",
            description="Tool server using Ray for distributed execution",
            version="1.0.0"
        )
        
        # Configure routes
        self._configure_app()
        
    def _configure_app(self):
        """Set up API routes"""
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        @self.app.post("/get_observation")
        async def handle_observation_request(request: Request):
            async with semaphore:
                # Parse request
                data = await request.json()
                
                try:
                    # Normalize trajectory IDs
                    if "trajectory_ids" in data:
                        data["trajectory_ids"] = [str(tid) if not isinstance(tid, str) else tid 
                                                for tid in data.get("trajectory_ids", [])]
                        
                    # Validate and process request
                    trajectory_ids = data.get("trajectory_ids", [])
                    actions = data.get("actions", [])
                    extra_keys = [k for k in data.keys() if k not in ["trajectory_ids", "actions"]]
                    extra_fields = [
                        {key: data[key][i] for key in extra_keys} 
                        for i in range(len(trajectory_ids))
                    ]
                    
                    # Process with Ray
                    observations, dones, valids = await self.tool_manager.process_actions(
                        trajectory_ids, actions, extra_fields
                    )
                    # import json
                    # with open("tool_results.json", 'w') as f:
                    #     json.dump({"observations": observations, "dones": dones, "valids": valids}, f, indent=4)
                    #     print(f"Results saved to tool_results.json")
                    return {
                        "observations": observations, 
                        "dones": dones, 
                        "valids": valids
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}", exc_info=True)
                    return {"error": str(e)}, 500
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "ray_status": "connected" if ray.is_initialized() else "disconnected"}
            
    def start(self):
        """Start the server"""
        logger.info(f"Starting Ray Tool Server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


# CLI entry point
def main(
    tool_type: Union[str, Tuple[str]] = "base",
    host: str = "0.0.0.0",
    port: int = 5000,
    workers_per_tool: int = 4,
    max_concurrent_requests: int = 64,
    ray_address: Optional[str] = None
):
    """
    Start the Ray Tool Server.
    
    Args:
        tool_type: Tool type(s) to initialize
        host: Host address
        port: Port number
        workers_per_tool: Number of Ray workers per tool type
        max_concurrent_requests: Maximum number of concurrent requests
        ray_address: Ray cluster address
    """
    # Initialize Ray
    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address)
        else:
            ray.init()
    
    # Convert tool_type to tuple if needed
    if isinstance(tool_type, str):
        if "," in tool_type:
            tool_type = tuple(t.strip() for t in tool_type.split(","))
        else:
            tool_type = (tool_type,)
    
    # Create and start server
    server = RayToolServer(
        tool_types=tool_type,
        host=host,
        port=port,
        workers_per_tool=workers_per_tool,
        max_concurrent_requests=max_concurrent_requests
    )
    server.start()


if __name__ == "__main__":
    import fire
    fire.Fire(main)