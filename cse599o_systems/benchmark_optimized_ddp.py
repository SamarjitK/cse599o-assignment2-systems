# benchmark_optimized_ddp.py
# -------------------------------------------------------------
# CSE 599O
#
# Extend your DDP benchmark to evaluate three optimized variants
# for the Transformer model:
#   (1) run_flat       
#   (2) run_individual 
#   (3) run_bucketed   
#
# The TA will execute your script using commands like:
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode flat
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode individual
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode bucketed --bucket-mb 10
#
# Each function should measure and print out the following statistics:
#   - iteration time per step  → append to iteration_times
#   - communication time per step → append to comm_times
# -------------------------------------------------------------

import argparse
import torch
import torch.distributed as dist
# Any other necessary imports can be added here.
import os
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
from cse599o_basics.adamw import AdamW
from cse599o_basics.model_utils import cross_entropy
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_systems.ddp import DDP, DDPBucketed

# Any necessary helper functions can be defined here.
def calculate_elapsed_time(start_events, end_events, num_iters):
    times = []
    for i in range(num_iters):
        time = start_events[i].elapsed_time(end_events[i])  # in milliseconds
        times.append(time)
    avg_time = sum(times) / num_iters
    return avg_time

# You can change the function and variable names as needed.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ============================================================
# (0) Naive DDP
# ============================================================
def naive_worker(rank, world_size, model, data, optimizer, num_iters, num_warmup, result_queue):
    setup(rank, world_size)

    model = model.to(f"cuda:{rank}")

    input, labels = data
    assert input.size(0) % world_size == 0, "d doesn't divide n"
    input = torch.chunk(input, world_size, dim=0)[rank].to(f"cuda:{rank}")
    labels = torch.chunk(labels, world_size, dim=0)[rank].to(f"cuda:{rank}")


    total_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    total_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    comm_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    comm_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters + num_warmup):
        iter = i - num_warmup
        if i >= num_warmup:
            total_starts[iter].record()
        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy(output, labels)
        loss.backward()

        # all-reduce to average loss gradients
        if i >= num_warmup:
            comm_starts[iter].record()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        if i >= num_warmup:
            comm_ends[iter].record()
        torch.cuda.synchronize()

        # going to divide separately so I'm sure it's not included in comm time
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= world_size

        optimizer.step()
        if i >= num_warmup:
            total_ends[iter].record()
        torch.cuda.synchronize()

    avg_total_time = calculate_elapsed_time(total_starts, total_ends, num_iters)
    avg_comm_time = calculate_elapsed_time(comm_starts, comm_ends, num_iters)
    result_queue.put((avg_total_time, avg_comm_time))
    cleanup()

# You can change the function and variable names as needed.
def run_naive(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """A naive DDP training loop for reference."""
    manager = mp.Manager()
    result_queue = manager.Queue()
    world_size = 2
    
    mp.spawn(
        naive_worker,
        args=(world_size, model, data, optimizer, num_iters, num_warmup, result_queue),
        nprocs=world_size,
        join=True,
    )

    for _ in range(world_size):
        iter_time, comm_time = result_queue.get()
        iteration_times.append(iter_time)
        comm_times.append(comm_time)


# ============================================================
# (1) Flat DDP
# ============================================================

def flat_worker(rank, world_size, model, data, optimizer, num_iters, num_warmup, result_queue):
    setup(rank, world_size)

    model = model.to(f"cuda:{rank}")

    input, labels = data
    assert input.size(0) % world_size == 0, "d doesn't divide n"
    input = torch.chunk(input, world_size, dim=0)[rank].to(f"cuda:{rank}")
    labels = torch.chunk(labels, world_size, dim=0)[rank].to(f"cuda:{rank}")


    total_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    total_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    comm_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    comm_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    total_elements = sum(param.numel() for param in model.parameters() if param.requires_grad)
    flat_grads = torch.zeros(total_elements, device=f"cuda:{rank}")

    for i in range(num_iters + num_warmup):
        iter = i - num_warmup
        if i >= num_warmup:
            total_starts[iter].record()
        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy(output, labels)
        if i >= num_warmup:
            nvtx.range_push("Computation")
        loss.backward()
        if i >= num_warmup:
            nvtx.range_pop()

        # all-reduce to average loss gradients
        offset = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.grad.data.numel()
                flat_grads[offset:offset + numel].copy_(param.grad.data.view(-1))
                offset += numel

        if i >= num_warmup:
            comm_starts[iter].record()
            nvtx.range_push("Communication")
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        if i >= num_warmup:
            nvtx.range_pop()
            comm_ends[iter].record()
        torch.cuda.synchronize()

        flat_grads /= world_size
        offset = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.grad.data.numel()
                param.grad.data.copy_(flat_grads[offset:offset + numel].view_as(param.grad.data))
                offset += numel

        optimizer.step()
        if i >= num_warmup:
            total_ends[iter].record()
        torch.cuda.synchronize()

    avg_total_time = calculate_elapsed_time(total_starts, total_ends, num_iters)
    avg_comm_time = calculate_elapsed_time(comm_starts, comm_ends, num_iters)
    result_queue.put((avg_total_time, avg_comm_time))
    cleanup()

# You can change the function and variable names as needed.
def run_flat(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce a single flattened gradient tensor."""
    manager = mp.Manager()
    result_queue = manager.Queue()
    world_size = 2
    
    mp.spawn(
        flat_worker,
        args=(world_size, model, data, optimizer, num_iters, num_warmup, result_queue),
        nprocs=world_size,
        join=True,
    )

    for _ in range(world_size):
        iter_time, comm_time = result_queue.get()
        iteration_times.append(iter_time)
        comm_times.append(comm_time)


# ============================================================
# (2) Individual DDP
# ============================================================
# You can change the function and variable names as needed.
def individual_worker(rank, world_size, model, data, optimizer, num_iters, num_warmup, result_queue):
    setup(rank, world_size)

    model = DDP(model.to(f"cuda:{rank}"))

    input, labels = data
    assert input.size(0) % world_size == 0, "d doesn't divide n"
    input = torch.chunk(input, world_size, dim=0)[rank].to(f"cuda:{rank}")
    labels = torch.chunk(labels, world_size, dim=0)[rank].to(f"cuda:{rank}")


    total_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    total_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters + num_warmup):
        iter = i - num_warmup
        if i >= num_warmup:
            total_starts[iter].record()
        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy(output, labels)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        if i >= num_warmup:
            total_ends[iter].record()
        torch.cuda.synchronize()

    avg_total_time = calculate_elapsed_time(total_starts, total_ends, num_iters)
    result_queue.put(avg_total_time)
    cleanup()

def run_individual(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce each parameter's gradient individually."""
    manager = mp.Manager()
    result_queue = manager.Queue()
    world_size = 2
    
    mp.spawn(
        individual_worker,
        args=(world_size, model, data, optimizer, num_iters, num_warmup, result_queue),
        nprocs=world_size,
        join=True,
    )

    for _ in range(world_size):
        iteration_times.append(result_queue.get())


# ============================================================
# (3) Bucketed DDP
# ============================================================

def bucketed_worker(rank, world_size, model, data, optimizer, num_iters, num_warmup, bucket_mb, result_queue):
    setup(rank, world_size)

    model = DDPBucketed(model.to(f"cuda:{rank}"), bucket_mb)

    input, labels = data
    assert input.size(0) % world_size == 0, "d doesn't divide n"
    input = torch.chunk(input, world_size, dim=0)[rank].to(f"cuda:{rank}")
    labels = torch.chunk(labels, world_size, dim=0)[rank].to(f"cuda:{rank}")


    total_starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    total_ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters + num_warmup):
        iter = i - num_warmup
        if i >= num_warmup:
            total_starts[iter].record()
        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy(output, labels)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        if i >= num_warmup:
            total_ends[iter].record()
        torch.cuda.synchronize()

    avg_total_time = calculate_elapsed_time(total_starts, total_ends, num_iters)
    result_queue.put(avg_total_time)
    cleanup()

# You can change the function and variable names as needed.
def run_bucketed(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, bucket_mb):
    """Group gradients into buckets and all-reduce each bucket."""
    manager = mp.Manager()
    result_queue = manager.Queue()
    world_size = 2
    
    mp.spawn(
        bucketed_worker,
        args=(world_size, model, data, optimizer, num_iters, num_warmup, bucket_mb, result_queue),
        nprocs=world_size,
        join=True,
    )

    for _ in range(world_size):
        iteration_times.append(result_queue.get())


# ============================================================
# Benchmark Function
# ============================================================
# You can change the function and variable names as needed.
def benchmark_optimized_ddp():
    """Benchmark DDP variants on the Transformer model."""
    parser = argparse.ArgumentParser(description="Benchmark optimized DDP variants.")
    parser.add_argument(
        "--mode",
        type=str,
        default="flat",
        choices=["flat", "individual", "bucketed", "naive"],
        help="Select which DDP variant to benchmark.",
    )
    parser.add_argument(
        "--bucket-mb",
        type=int,
        default=10,
        help="Bucket size (in MB) for the bucketed DDP variant.",
    )
    args = parser.parse_args()

    # Example placeholders
    num_iters, num_warmup = 5, 2
    iteration_times, comm_times = [], []
    
    # DDP setup
    # TODO: Initialize distributed process group
    mp.set_start_method("spawn", force=True)
    world_size = 2

    # Construct model and move to GPU
    # TODO: Define model parameters
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
    model = TransformerLM(**model_args)
    

    # Construct optimizer
    # TODO: Define optimizer
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = AdamW(model.parameters(), **optim_args)
    
    # Dummy data
    # TODO: Create input data
    batch_size = 4 * world_size
    seq_length = model_args["context_length"]
    data = ( # input, labels
        torch.randint(0, vocab_size, (batch_size, seq_length)),
        torch.randint(0, vocab_size, (batch_size, seq_length))
    )

    if args.mode == "naive":
        run_naive(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "flat":
        run_flat(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "individual":
        run_individual(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "bucketed":
        run_bucketed(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, args.bucket_mb)

    print(f"Mode: {args.mode}")
    print(f"Iteration times: {iteration_times}")
    print(f"Communication times: {comm_times}")

    print(f"Average percentage of time spent in communication: {sum(comm_times)/sum(iteration_times)*100:.2f} %")

if __name__ == "__main__":
    benchmark_optimized_ddp()
