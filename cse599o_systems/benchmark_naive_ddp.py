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
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cse599o_basics.adamw import AdamW
from cse599o_basics.model_utils import cross_entropy
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from tests.common import ToyModel

# Any necessary helper functions can be defined here.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Initialize NCCL for GPU-based distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# You can change the function and variable names as needed.
def run_naive_ddp_worker(rank, world_size, data, num_steps, result_queue, model, model_params =None):
    """Run one DDP worker process."""
    setup(rank, world_size)

    # each device constructs random model
    model = model.to(f"cuda:{rank}")
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }

    if rank == 0 and model_params: # this is just for verification, since we need to match baseline
        model.load_state_dict(model_params)

    # send params from rank 0 to all others
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    optim = AdamW(model.parameters(), **optim_args)

    input, labels = data
    assert input.size(0) % world_size == 0, "d doesn't divide n"
    input = torch.chunk(input, world_size, dim=0)[rank].to(f"cuda:{rank}")
    labels = torch.chunk(labels, world_size, dim=0)[rank].to(f"cuda:{rank}")

    # need to measure total training time per iteration
    # and fraction of time spent on gradient communication
    total_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
    total_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
    comm_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
    comm_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]

    for i in range(num_steps):
        total_starts[i].record()
        optim.zero_grad()
        output = model(input)
        loss = cross_entropy(output, labels)
        loss.backward()

        # all-reduce to average loss gradients
        comm_starts[i].record()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        comm_ends[i].record()
        torch.cuda.synchronize()

        # going to divide separately so I'm sure it's not included in comm time
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= world_size

        optim.step()
        total_ends[i].record()
        torch.cuda.synchronize()

    if model_params:
        if rank == 0:
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            result_queue.put(state)
    else: # we're doing timing benchmark
        total_times = []
        comm_times = []
        torch.cuda.synchronize()
        for i in range(num_steps):
            total_time = total_starts[i].elapsed_time(total_ends[i])  # in milliseconds
            comm_time = comm_starts[i].elapsed_time(comm_ends[i])  # in milliseconds
            total_times.append(total_time)
            comm_times.append(comm_time)
        avg_total_time = sum(total_times) / num_steps
        avg_comm_time = sum(comm_times) / num_steps
        result_queue.put((avg_total_time, avg_comm_time))
    cleanup()

# You can change the function and variable names as needed.
def run_baseline(data, num_steps, model_params):
    """Run single-process baseline for comparison."""
    model = ToyModel()
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    model.load_state_dict(model_params)
    optim = AdamW(model.parameters(), **optim_args)

    input, labels = data
    for _ in range(num_steps):
        optim.zero_grad()
        output = model(input)
        loss = cross_entropy(output, labels)
        loss.backward()
        optim.step()

    return model.state_dict()

# You can change the function and variable names as needed.
def verify_naive_ddp():
    """Benchmark and verify naive DDP."""
    world_size = 2
    num_steps = 5
    data = torch.randn(10, 10), torch.randint(0, 5, (10,))

    # create model and optim bc otherwise the random init will differ
    model = ToyModel()
    model_params = {k: v.clone() for k, v in model.state_dict().items()}

    # Run baseline
    no_ddp_state = run_baseline(data, num_steps, model_params)

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_naive_ddp_worker,
        args=(world_size, data, num_steps, result_queue, model, model_params),
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
    # make dummy params
    tokenizer = BPETokenizer(vocab={}, merges=[])
    vocab_size = tokenizer.tokenizer.n_vocab
    model_args = {
        "vocab_size": vocab_size,
        "context_length": 256,
        "num_layers": 36,
        "d_model": 1280,
        "num_heads": 20,
        "d_ff": 5120,
        "rope_theta": 10000.0
    }
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }

    world_size = 2
    num_steps = 10
    batch_size = 4 * world_size
    seq_length = model_args["context_length"]
    data = ( # input, labels
        torch.randint(0, vocab_size, (batch_size, seq_length)),
        torch.randint(0, vocab_size, (batch_size, seq_length))
    )

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    model = TransformerLM(**model_args)
    mp.spawn(
        run_naive_ddp_worker,
        args=(world_size, data, num_steps, result_queue, model, None),
        nprocs=world_size,
        join=True,
    )

    # each process will put its timing results in the queue
    total_times = []
    comm_times = []
    for _ in range(world_size):
        avg_total_time, avg_comm_time = result_queue.get()
        # also print raw data for testing
        print(f"GPU {_}: Average Step Time: {avg_total_time:.2f} ms, Average Comm Time: {avg_comm_time:.2f} ms")
        total_times.append(avg_total_time)
        comm_times.append(avg_comm_time)
    print(f"Naive DDP Transformer Benchmark over {num_steps} steps with {world_size} GPUs:")
    print(f"Average Step Time per GPU: {sum(total_times)/world_size:.2f} ms")
    print(f"Average Communication Time per GPU: {sum(comm_times)/world_size:.2f} ms")
    print(f"Fraction of Time in Communication: {sum(comm_times)/sum(total_times)*100:.2f} %")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["toy", "transformer"], default="toy")
    args = parser.parse_args()

    if args.model == "toy":
        verify_naive_ddp()
    elif args.model == "transformer":
        timing_naive_ddp()