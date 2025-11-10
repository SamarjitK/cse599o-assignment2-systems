# benchmark_naive_ddp.py
# -------------------------------------------------------------
# CSE 599O: Distributed Training Basics
#
# Implement a naive DDP version that reproduces the same model
# state as single-process training.
#
# The TA will test your implementation with the following commands:
#
# 1. To verify that DDP matches baseline (toy model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model toy
# Expected output: "Naive DDP matches baseline!"
#
# 2. To output communication and step time (transformer model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model transformer
# Expected output: communication and step time statistics
#
# -------------------------------------------------------------

# Any necessary imports can be added here.
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cse599o_basics.adamw import AdamW
from tests.common import ToyModel

# Any necessary helper functions can be defined here.

# You can change the function and variable names as needed.
def run_naive_ddp_worker(rank, world_size, data, num_steps, result_queue):
    """Run one DDP worker process."""
    
    if rank == 0:
        # TODO: Collect and return the model state from rank 0
        result_queue.put({})  # Replace with actual model state

# You can change the function and variable names as needed.
def run_baseline(data, num_steps):
    """Run single-process baseline for comparison."""
    model = ToyModel()
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optim = AdamW(model.parameters(), **optim_args)
    for _ in range(num_steps):
        optim.zero_grad()
        output = model(data)
        loss = output.sum()
        loss.backward()
        optim.step()

    return model.state_dict()

# You can change the function and variable names as needed.
def verify_naive_ddp():
    """Benchmark and verify naive DDP."""
    world_size = 2
    num_steps = 5
    data = torch.randn(10, 5)

    # Run baseline
    no_ddp_state = run_baseline(data, num_steps)

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_naive_ddp_worker,
        args=(world_size, data, num_steps, result_queue),
        nprocs=world_size,
        join=True,
    )

    # Get model state from DDP
    ddp_state = result_queue.get()
    
    assert len(no_ddp_state) > 0, "model state from baseline is empty"
    for name in no_ddp_state:
        assert torch.allclose(no_ddp_state[name], ddp_state[name], atol=1e-6)
    print("Naive DDP matches baseline!")
  
# You can change the function and variable names as needed.  
def timing_naive_ddp():
    """Timing benchmark for naive DDP with transformer model."""
    # TODO
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["toy", "transformer"], default="toy")
    args = parser.parse_args()

    if args.model == "toy":
        verify_naive_ddp()
    elif args.model == "transformer":
        timing_naive_ddp()