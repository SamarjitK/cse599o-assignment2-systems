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
from multiprocessing import Manager
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.adamw import AdamW
from cse599o_basics.model_utils import cross_entropy
from timeit import default_timer as timer
# You can add other necessary imports here.


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs):
        param_groups = list(params)
        self.params = param_groups
        param_groups = [{"params": self.params}]

        dummy_params = [{"params": []}]
        self.optimizer = optimizer_cls(dummy_params, **kwargs)
        super().__init__(param_groups, {})
        self.optimizer.param_groups.pop(0)

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)
        self.synchronize_params()

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        param_group['params'] = self.sharded_params(param_group['params'], rank, world_size)
        self.optimizer.add_param_group(param_group)
        super().add_param_group(param_group)

    def sharded_params(self, params, rank, world_size):
        return [ p for i, p in enumerate(params) if i % world_size == rank ]

    def synchronize_params(self):
        world_size = dist.get_world_size()
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.requires_grad:
                    dist.broadcast(param.data, src= i % world_size)

# Add any necessary helper functions here.
def calculate_elapsed_time(start_events, end_events, num_iters):
    times = []
    for i in range(num_iters):
        time = start_events[i].elapsed_time(end_events[i])  # in milliseconds
        times.append(time)
    avg_time = sum(times) / num_iters
    return avg_time

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# You can change the function and variable names as needed.
def run_distributed_training(rank, world_size, num_trials, num_warmup_trials, result_queue):
    # Setup distributed environment
    setup(rank, world_size)

    # Construct model
    # add an nvtx marker to check memory usage right after model initialization
    vocab_size = 5
    model_args = {
        "vocab_size": vocab_size,
        "context_length": 4,
        "num_layers": 36,
        "d_model": 1280,
        "num_heads": 20,
        "d_ff": 5120,
        "rope_theta": 10000.0
    }
    model = TransformerLM(**model_args).to(f"cuda:{rank}")
    mem = torch.cuda.memory.max_memory_allocated()
    print(f"After model init: Max memory allocated: {mem / (1024**2):.2f} MB")
    

    # Construct random input data
    batch_size = 4 * world_size
    seq_length = model_args["context_length"]
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length)).to(f"cuda:{rank}")
    labels = torch.randint(0, vocab_size, (batch_size, seq_length)).to(f"cuda:{rank}")

    # Construct optimizer
    # You can use the SharedOptimizer here
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = ShardedOptimizer(model.parameters(), AdamW, **optim_args)
    # optimizer = AdamW(model.parameters(), **optim_args)
    
    # timer stuff
    total_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    total_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    
    for i in range(num_trials + num_warmup_trials):
        iter = i - num_warmup_trials
        if i >= num_warmup_trials:
            total_starts[iter].record()
        optimizer.zero_grad()
        output = model(inputs)
        loss = cross_entropy(output, labels)
        loss.backward()
        if i == 0:
            mem = torch.cuda.memory.max_memory_allocated()
            print(f"Before step: Max memory allocated: {mem / (1024**2):.2f} MB")
        optimizer.step()
        if i == 0:
            mem = torch.cuda.memory.max_memory_allocated()
            print(f"After step: Max memory allocated: {mem / (1024**2):.2f} MB")
        if i >= num_warmup_trials:
            total_ends[iter].record()
        torch.cuda.synchronize()

    avg_total_time = calculate_elapsed_time(total_starts, total_ends, num_trials)
    result_queue.put(avg_total_time)

    cleanup()

if __name__ == "__main__":
    # Set up distributed training parameters
    # Collect results and print timing summary
    mp.set_start_method("spawn", force=True)
    world_size = 2
    num_iters, num_warmup = 5, 2

    manager = mp.Manager()
    result_queue = manager.Queue()
    world_size = 2
    
    mp.spawn(
        run_distributed_training,
        args=(world_size, num_iters, num_warmup, result_queue),
        nprocs=world_size,
        join=True,
    )

    iteration_times = []

    for _ in range(world_size):
        iteration_times.append(result_queue.get())
    
    print(f"Times: {iteration_times}")