from dataclasses import dataclass

@dataclass
class AgentActorConfig:
    max_turns: int=0
    max_start_length: int=None
    max_prompt_length: int=None
    max_response_length: int=None
    max_obs_length: int=None
    tokenizer_path: str=None
    # logging: dict
    num_gpus: int=1
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    tool_server_url: str = None