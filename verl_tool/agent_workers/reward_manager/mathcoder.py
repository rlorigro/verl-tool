# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import json
import hashlib
import random
import os
import json
import subprocess
import time
import regex as re
from pathlib import Path
import uuid

import torch
from collections import defaultdict
from verl import DataProto
from verl.protocol import collate_fn
from verl_tool.agent_workers.reward_manager.reward_score import _default_compute_score
from verl_tool.agent_workers.reward_manager.reward_score.torl_math import (
    compute_score as torl_compute_score,
)

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

from .torl import ToRLRewardManager
from .acecoder import AceCoderRewardManager
class MathCoderRewardManager:
    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"
    ) -> None:
        self.ToRLRewardManager = ToRLRewardManager(
            tokenizer, num_examine, compute_score, reward_fn_key
        )
        self.AceCoderRewardManager = AceCoderRewardManager(
            tokenizer, num_examine, compute_score
        )

    def __call__(self, data: DataProto, return_dict=False):
        data.batch['reward_id'] = torch.arange(len(data), device=data.batch.device) # TODO: check here
        code_data = collate_fn(
            [x for x in data if x.non_tensor_batch["ability"] == "code"]
        ) 
        math_data = collate_fn(
            [x for x in data if x.non_tensor_batch["ability"] == "math"]
        )
        code_reward = self.AceCoderRewardManager(code_data, return_dict=True)  
        math_reward = self.ToRLRewardManager(math_data, return_dict=True)
        # TODO: delete
        print("len code_reward", len(code_reward['reward_tensor']))
        print("len math_reward", len(math_reward['reward_tensor']))
        reward_tensor = torch.cat([code_reward['reward_tensor'], math_reward['reward_tensor']])
        reward_id = torch.cat([code_reward['reward_id'], math_reward['reward_id']])
        _, indices = torch.sort(reward_id)
        sorted_reward = reward_tensor[indices]
        
        if return_dict:
            return {
                "reward_tensor": sorted_reward,
            }
        else:
            return sorted_reward 