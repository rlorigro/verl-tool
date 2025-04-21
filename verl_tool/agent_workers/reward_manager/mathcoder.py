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
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # reward extra info every key of it is a default len(data) list filled with None
        reward_extra_info = defaultdict(
            lambda: [None] * len(data)
        )
        
        code_data_idxs = [
            i for i in range(len(data)) if data[i].non_tensor_batch["ability"] == "code"
        ]
        math_data_idxs = [
            i for i in range(len(data)) if data[i].non_tensor_batch["ability"] == "math"
        ]
        code_data = data[code_data_idxs]
        math_data = data[math_data_idxs]
        code_reward = self.AceCoderRewardManager(code_data, return_dict=True)
        math_reward = self.ToRLRewardManager(math_data, return_dict=True)
        print("len code_reward", len(code_reward['reward_tensor']))
        print("len math_reward", len(math_reward['reward_tensor']))
        # put the code and math reward together in the original order
        reward_tensor[code_data_idxs] = code_reward['reward_tensor']
        reward_tensor[math_data_idxs] = math_reward['reward_tensor']
        
        for k, v in code_reward['reward_extra_info'].items():
            if k not in reward_extra_info:
                for i in range(len(v)):
                    reward_extra_info[f"code_{k}"][code_data_idxs[i]] = v[i]
            re
        for k, v in math_reward['reward_extra_info'].items():
            if k not in reward_extra_info:
                for i in range(len(v)):
                    reward_extra_info[f"math_{k}"][math_data_idxs[i]] = v[i]
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor