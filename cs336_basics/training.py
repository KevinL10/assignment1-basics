import os
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import cross_entropy, get_batch, load_checkpoint, save_checkpoint
import numpy as np
import torch
import wandb

from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import Transformer


def train(
    dataset_path: str | os.PathLike,
    batch_size: int = 32,
    context_length: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
    checkpoint_dir: str | os.PathLike | None = None,
):
    """Main training loop.

    Args:
        dataset_path: path to the numpy array containing the tokenized training dataset
        ckpt_path: path to save intermediate model checkpoints
    """

    wandb.init(
        # name="basic",
        project="cs336-assignment1",
        config={"batch_size": batch_size, "context_length": context_length, "lr": lr},
    )

    model = Transformer(
        d_model=128,
        num_heads=4,
        d_ff=512,
        vocab_size=10_000,
        context_length=context_length,
        num_layers=4,
        theta=10_000,
    )
    opt = AdamW(model.parameters(), lr=lr)

    dataset = np.load(dataset_path, mmap_mode="r")

    model.train()
    step = 0
    while step < 1000:
        opt.zero_grad()

        x, y = get_batch(dataset, batch_size, context_length, device)
        loss = cross_entropy(model(x), y)
        loss.backward()
        opt.step()

        wandb.log({"loss": loss.item()})
        print("loss at step", step, loss.item())
        step += 1

        if step % 100 == 0 and checkpoint_dir is not None:
            print("saving checkpoint at step", step)
            save_checkpoint(model, opt, step, f"{checkpoint_dir}/checkpoint_{step}.pt")


def predict():
    torch.manual_seed(1337)
    model = Transformer(
        d_model=128,
        num_heads=4,
        d_ff=512,
        vocab_size=10_000,
        context_length=256,
        num_layers=4,
        theta=10_000,
    )
    opt = AdamW(model.parameters(), lr=1e-3)

    load_checkpoint("data/checkpoints/checkpoint_600.pt", model, opt)

    tokenizer = Tokenizer.from_files("data/tinystories_vocab.json", "data/tinystories_merges.txt", "<|endoftext|>")
    print(len(tokenizer.vocab), len(tokenizer.rank))
    text = "There was "

    tokens = tokenizer.encode(text)
    out = model.generate(tokens, temperature=0.0, top_p=0.9)
    print(tokenizer.decode(out))

    tokens = tokenizer.encode(text)
    out = model.generate(tokens, temperature=0.0, top_p=0.9)
    print(tokenizer.decode(out))


if __name__ == "__main__":
    train("data/tinystories_encoded.npy", checkpoint_dir="data/checkpoints")
    # predict()
