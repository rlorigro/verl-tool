"""
Metrics related to the Agent PPO trainer. Change it to add more metrics.
"""

import sys

import verl.trainer.ppo.metric_utils
verl_computer_data_metrics = verl.trainer.ppo.metric_utils.compute_data_metrics

import torch
from typing import Any, Dict, List
import numpy as np
from verl import DataProto

def agent_compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    metrics = verl_computer_data_metrics(batch, use_critic)
     # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    print(metrics)
    return metrics

sys.modules["verl.trainer.ppo.metric_utils"].compute_data_metrics = agent_compute_data_metrics