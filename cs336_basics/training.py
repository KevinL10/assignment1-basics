import os
from cs336_basics.utils import cross_entropy, get_batch
import numpy as np

from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import Transformer


def train(
    dataset_path: str | os.PathLike,
    batch_size: int = 32,
    context_length: int = 256,
    device: str | None = None,
    ckpt_path: str | os.PathLike | None = None,
):
    """Main training loop.

    Args:
        dataset_path: path to the numpy array containing the tokenized training dataset
        ckpt_path: path to save intermediate model checkpoints
    """

    model = Transformer(
        d_model=128,
        num_heads=4,
        d_ff=512,
        vocab_size=10_000,
        context_length=context_length,
        num_layers=4,
        theta=10_000,
    )
    opt = AdamW(model.parameters(), lr=1e-3)

    dataset = np.load(dataset_path, mmap_mode="r")

    model.train()
    step = 0
    while step < 100:
        opt.zero_grad()

        x, y = get_batch(dataset, batch_size, context_length, device)
        loss = cross_entropy(model(x), y)
        loss.backward()
        opt.step()

        print(f"loss at step {step}", loss)
        step += 1


if __name__ == "__main__":
    train("data/tinystories_encoded.npy")
