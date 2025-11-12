# sharding_optimizer.py
# -------------------------------------------------------------
# CSE 599O: 
#
# Implement optimizer state sharding for distributed training.
#
# -------------------------------------------------------------
import os
from typing import Dict
import torch
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
from typing import Any, Type
from torch.optim import Optimizer
from cse599o_basics.adamw import AdamW
from multiprocessing import Manager
from timeit import default_timer as timer
# You can add other necessary imports here.


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs):
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("params were empty")
        self.optimizer = optimizer_cls(params, **kwargs)
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("params became empty")
        super().__init__(params, self.optimizer.defaults)
    
    def step(self, closure=None, **kwargs):
        print(f"param groups: {self.param_groups}")
        self.optimizer.step(closure, **kwargs)
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    for state_key, state_value in self.optimizer.state[param].items():
                        if torch.is_tensor(state_value):
                            dist.all_reduce(state_value, op=dist.ReduceOp.SUM)
                            state_value /= dist.get_world_size()

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        print(f"add_param_group called from: {self.__class__.__name__}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        param_group['params'] = self.sharded_params(param_group['params'], rank, world_size)
        super().add_param_group(param_group)

    def sharded_params(self, params, rank, world_size):
        sharded_params = []
        for i, param in enumerate(params):
            if i % world_size == rank:
                sharded_params.append(param)
        return sharded_params

# Add any necessary helper functions here.

# You can change the function and variable names as needed.
def run_distributed_training(rank, world_size, num_trials, num_warmup_trials, result_queue):
    # Setup distributed environment
    # TODO

    # Construct model
    # TODO

    # Construct random input data
    # TODO: Create input data

    # Construct optimizer
    # You can use the SharedOptimizer here
    # TODO
    
    # Training loop
        # Warm up
        # TODO
        # Benchmark
        # TODO
    
    if rank == 0:
        pass
       # Collect and return the timing results

if __name__ == "__main__":
    # Set up distributed training parameters
    # Collect results and print timing summary
    print("nothing")