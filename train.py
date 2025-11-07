import argparse
import os
import torch
import json
import numpy as np

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.adamw import AdamW
from cse599o_basics.train_utils import prep_datasets, train

def main():
    train_txt, valid_txt = "data/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-valid.txt"
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    # transformerLM hyperparameters
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # get_batch parameters
    parser.add_argument("--batch_size", type=int, default=32)

    # training parameters
    parser.add_argument("--num_steps", type=int, default=5000)

    args = parser.parse_args()
    context_length: int = args.context_length
    num_layers: int = args.num_layers
    d_model: int = args.d_model
    num_heads: int = args.num_heads
    d_ff: int = args.d_ff
    rope_theta: float = args.rope_theta
    batch_size: int = args.batch_size
    num_steps: int = args.num_steps

    print("Initializing tokenizer...")
    tokenizer = BPETokenizer(vocab={}, merges=[])
    vocab_size = tokenizer.tokenizer.n_vocab

    print("Preparing datasets...")
    train_dataset, valid_dataset = prep_datasets(train_txt, valid_txt, tokenizer)

    print("Initializing model and optimizer...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_args = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta
    }
    optim_args = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    model = TransformerLM(**model_args).to(device)
    optim = AdamW(model.parameters(), **optim_args)

    print("Starting training loop...")
    results = train(model, optim, train_dataset, valid_dataset,
                    batch_size, context_length, num_steps, device,
                    # lr_scheduler_params=[optim_args["lr"], 1e-4, int(0.25 * num_steps), num_steps],
                    # max_grad_norm=1.0
                    )
    results.update({
            "model_args": model_args,
            "optim_args": optim_args,
            "batch_size": batch_size,
            "num_steps": num_steps,
        })
    report_file = os.path.join("reports", f"report_{results['timestamp']}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved report to {report_file}")

if __name__ == "__main__":
    main()