import torch
import torch.distributed as dist

class DDPBucketed(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, bucket_size_mb: float):
        super(DDPBucketed, self).__init__()
        self.module = model
        current_bucket = 0
        current_bucket_size = 0.0
        for param in reversed(list(model.parameters())):
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param_size = param.numel() * param.element_size() / (1024 * 1024)  # size in MB
                if current_bucket_size + param_size > bucket_size_mb:
                    current_bucket += 1
                    current_bucket_size = 0.0
                current_bucket_size += param_size
                param.register_post_accumulate_grad_hook(lambda p, b=current_bucket: self._hook(p, b))
        self.buckets_handles = [[] for _ in range(current_bucket + 1)]

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handles in self.buckets_handles:
            for handle in handles:
                handle.wait()
            handles.clear()
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.data /= world_size

    def _hook(self, param, bucket_id):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.buckets_handles[bucket_id].append(handle)

class DDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(DDP, self).__init__()
        self.module = model
        self.handles = []
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._hook)
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.data /= world_size

    def _hook(self, param):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
