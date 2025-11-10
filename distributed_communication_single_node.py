import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Initialize NCCL for GPU-based distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def distributed_demo(rank, world_size, size):
    setup(rank, world_size)
    data = torch.randn(size, device=f"cuda:{rank}")

    # warm ups
    for _ in range(5):
        dist.all_reduce(data)
    dist.barrier()

    # timed all-reduce
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record(torch.cuda.current_stream())
    dist.all_reduce(data)
    end.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)

    # gathering
    results = [0.0 for _ in range(world_size)]
    dist.all_gather_object(results, elapsed)
    if rank == 0:
        avg_time = sum(results) / len(results)
        print(f"Average for {world_size} GPUs, size {size * 4 / (1024 * 1024)} MB: {avg_time} ms")
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() # auto-detect GPUs
    # sizes: 1 MB, 10 MB, 100 MB, 1 GB, float32 = 4 bytes so divide by 4
    sizes = [1024 * 1024 * (10 ** i) // 4 for i in range(4)]
    if world_size < 2:
        raise RuntimeError("Need at least 2 GPUs to run this example.")
    for size in sizes:
        mp.spawn(distributed_demo, args=(world_size, size), nprocs=world_size, join=True)