import torch
import numpy.typing as npt
import os
import timeit
import numpy as np
from typing import BinaryIO, IO
from cse599o_basics.model_utils import cross_entropy, softmax, lr_scheduler, gradient_clipping
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.adamw import AdamW
from cse599o_basics.tokenizer import BPETokenizer

def decode(model: TransformerLM, tokenizer: BPETokenizer, optim: AdamW,
           max_tokens:int = 256, temperature: float = 1.0, top_p: float = 0.9, 
           prompt: str = "Once upon a time", ckpt_file: str = "") -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if ckpt_file and os.path.exists(ckpt_file):
        print("Loading checkpoint...")
        load_checkpoint(ckpt_file, model, optim)
    else:
        print("No checkpoint. I sure hope you've trained this model!")

    print("Generating text...")
    model.eval()
    input_tokens = torch.tensor(tokenizer.encode(prompt)) # (seq_len,)
    input_tensor = input_tokens.unsqueeze(0).to(device) # (1, seq_len)
    eot = tokenizer.tokenizer.eot_token

    # keep list of generated tokens
    generated_tokens = input_tokens.tolist()

    with torch.no_grad():
        for _ in range(max_tokens - input_tokens.size(0)):
            outputs = model(input_tensor) # (1, seq_len, vocab_size)
            last_token_softmax = softmax(outputs[0, -1, :] / temperature, dim=-1) # (vocab_size,)
            sorted_probs, sorted_indices = torch.sort(last_token_softmax, descending=True)
            sum = 0.0
            i = 0
            while sum < top_p and i < sorted_probs.size(0):
                sum += sorted_probs[i].item()
                i += 1
            top_p_indices = sorted_indices[:i]
            top_p_probs = sorted_probs[:i]
            top_p_probs = top_p_probs / top_p_probs.sum()

            index = torch.multinomial(top_p_probs, 1)
            next_token = top_p_indices[index] # (1,)
            generated_tokens.append(next_token.item())
            if next_token.item() == eot:
                break
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0).to(device)], dim=1)

    return tokenizer.decode(generated_tokens)

def prep_datasets(train_txt: str, valid_txt: str, tokenizer: BPETokenizer):
    if not os.path.exists("data/train_memmap.dat"):
        print("Tokenizing training data...")
        with open(train_txt, "r", encoding="utf-8") as f:
            train_data = f.read()
        train_tokens = tokenizer.encode(train_data)
        print("Creating training memmap dataset...")
        train_dataset = np.memmap("data/train_memmap.dat", dtype=np.uint16, mode="w+", shape=(len(train_tokens),))
        train_dataset[:] = np.array(train_tokens, dtype=np.uint16)
        print("Repeating for validation data...")
        with open(valid_txt, "r", encoding="utf-8") as f:
            valid_data = f.read()
        valid_tokens = tokenizer.encode(valid_data)
        valid_dataset = np.memmap("data/valid_memmap.dat", dtype=np.uint16, mode="w+", shape=(len(valid_tokens),))
        valid_dataset[:] = np.array(valid_tokens, dtype=np.uint16)
    else:
        train_dataset = np.memmap("data/train_memmap.dat", dtype=np.uint16, mode="r")
        valid_dataset = np.memmap("data/valid_memmap.dat", dtype=np.uint16, mode="r")
    return train_dataset, valid_dataset

def train(model: TransformerLM, optim: AdamW, 
          train_dataset: np.memmap, valid_dataset: np.memmap,
          batch_size, context_length, num_steps, device,
          lr_scheduler_params: list = [], max_grad_norm: float | None = None) -> dict:
    timestamp = int(timeit.default_timer())
    ckpt_file = os.path.join("checkpoints", f"ckpt_{timestamp}.pt")
    train_losses = []
    val_losses = []
    step_times = []
    val_interval = num_steps // 20 if num_steps >= 20 else 2
    # inputs, labels = get_batch(train_dataset, batch_size, context_length, device) # to overfit
    for step in range(1, num_steps + 1):
        start = timeit.default_timer()
        model.train()

        # Learning rate scheduling
        if lr_scheduler_params:
            lr = lr_scheduler((step-1), *lr_scheduler_params)
            for group in optim.param_groups:
                group['lr'] = lr

        inputs, labels = get_batch(train_dataset, batch_size, context_length, device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)
        loss.backward()

        # Gradient clipping
        if max_grad_norm is not None and step > float(0.5 * num_steps):
            gradient_clipping(model.parameters(), max_grad_norm)
        
        optim.step()

        train_losses.append(loss.item())
        step_times.append(timeit.default_timer() - start)
        if (num_steps < 50 and step % 2 == 0)  or (num_steps >= 50 and step % (num_steps // 50) == 0):
            print(f"Step {step}, Training Loss: {loss.item():.4f}")

        if (num_steps < 20 and step % 2 == 0) or (num_steps >= 20 and step % (num_steps // 20) == 0):
            model.eval()
            with torch.no_grad():
                val_inputs, val_labels = get_batch(valid_dataset, batch_size, context_length, device)
                val_outputs = model(val_inputs)
                val_loss = cross_entropy(val_outputs, val_labels)
                val_losses.append(val_loss.item())
                print(f"Step {step}, Validation Loss: {val_loss.item():.4f}")
            save_checkpoint(model, optim, step, ckpt_file)
    # save_checkpoint(model, optim, num_steps, ckpt_file)
    return {"train_losses": train_losses, "val_losses": val_losses, "val_interval": val_interval,
            "ckpt_file": ckpt_file, "timestamp": timestamp}

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    states: dict = torch.load(src)
    model.load_state_dict(states['model_state_dict'])
    optimizer.load_state_dict(states['optimizer_state_dict'])
    return states['iteration']

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    """Dump all state from model, optimizer, and iteration into `out`."""
    # Use state_dict() for both model and optimizer
    # Use torch.save(obj, out) to dump obj into out
    states = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(states, out)

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]
    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    labels = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    for i in range(batch_size):
        start = torch.randint(0, n - context_length, (1,)).item()
        inp = torch.tensor(dataset[start : start + context_length], dtype=torch.long, device=device)
        lab = torch.tensor(dataset[start + 1 : start + context_length + 1], dtype=torch.long, device=device)
        inputs[i].copy_(inp)
        labels[i].copy_(lab)
    return inputs, labels