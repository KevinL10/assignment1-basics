import math
import os
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import (
    clip_gradients,
    cross_entropy,
    get_batch,
    load_checkpoint,
    lr_cosine_schedule,
    save_checkpoint,
)
import numpy as np
import torch
import wandb
import dataclasses
import argparse

from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import Transformer


@dataclasses.dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    d_ff: int
    vocab_size: int
    context_length: int
    num_layers: int
    theta: float | None = None


def train(
    train_dataset_path: str | os.PathLike,
    val_dataset_path: str | os.PathLike,
    transformer_config: TransformerConfig,
    batch_size: int = 32,
    lr: float = 1e-3,
    # total_tokens: int = 327_680_000,
    total_tokens: int = 40_000_000,
    validation_batches: int = 4,
    device: str | None = None,
    checkpoint_dir: str | os.PathLike | None = None,
):
    """Main training loop.

    Args:
        dataset_path: path to the numpy array containing the tokenized training dataset
        ckpt_path: path to save intermediate model checkpoints
    """

    wandb.init(
        name=f"transformer_lr{lr}_bs{batch_size}",
        project="cs336-assignment1",
        config={"batch_size": batch_size, "lr": lr, **dataclasses.asdict(transformer_config)},
    )
    print(f"Initialized wandb run with {lr=} and {batch_size=}")

    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, wandb.run.name + "_" + wandb.run.id)
        os.makedirs(checkpoint_path, exist_ok=True)

    model = Transformer(**dataclasses.asdict(transformer_config)).to(device)
    model = torch.compile(model, backend="aot_eager" if device == "mps" else "inductor")
    opt = AdamW(model.parameters(), lr=lr)

    train_dataset = np.load(train_dataset_path, mmap_mode="r")
    val_dataset = np.load(val_dataset_path, mmap_mode="r")

    print("train_dataset length:", len(train_dataset))
    print("val_dataset length:", len(val_dataset))

    model.train()
    max_steps = total_tokens // (batch_size * transformer_config.context_length)
    print("running for", max_steps, "steps")

    step = 0
    while step < max_steps:
        opt.zero_grad()

        x, y = get_batch(train_dataset, batch_size, transformer_config.context_length, device)
        loss = cross_entropy(model(x), y)
        loss.backward()

        clip_gradients(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr_cosine_schedule(step, lr, lr / 100, int(max_steps * 0.02), max_steps)

        opt.step()

        wandb.log(
            {"train_loss": loss.item(), "ppl": math.exp(loss.item()), "lr": opt.param_groups[0]["lr"]},
            step=step,
        )
        print("loss at step", step, loss.item())
        step += 1

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for _ in range(validation_batches):
                    x, y = get_batch(val_dataset, batch_size, transformer_config.context_length, device)
                    loss = cross_entropy(model(x), y)
                    val_loss += loss.item()

                val_loss /= validation_batches

            wandb.log({"val_loss": val_loss}, step=step)
            model.train()

        if step % (max_steps // 5) == 0 and checkpoint_dir is not None:
            print("saving checkpoint at step", step)
            save_checkpoint(model, opt, step, f"{checkpoint_path}/checkpoint_{step}.pt")

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    default_config = TransformerConfig(
        d_model=512,
        num_heads=16,
        d_ff=1344,
        vocab_size=10_000,
        context_length=256,
        num_layers=4,
        theta=10_000,
    )

    train(
        "data/tinystories_train.npy",
        "data/tinystories_validation.npy",
        default_config,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir="data/checkpoints",
        device="mps",
    )


if __name__ == "__main__":
    main()
