from typing import IO, BinaryIO, Iterable
import os
from cs336_basics.transformer import softmax
import torch
import math
import numpy.typing as npt
import numpy as np


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Returns the average cross-entropy across the batch."""
    # subtract max of elements for numerical stability
    logits = inputs - torch.amax(inputs, dim=-1, keepdim=True)
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    nll = -log_probs.gather(-1, targets.unsqueeze(-1))
    return nll.mean()


def lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """Returns the learning rate at the given iteration."""

    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    if warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 1 / 2 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    """Clips gradients of parameters in-place"""

    norm = 0.0
    for p in parameters:
        if p.grad is not None:
            norm += p.grad.detach().norm(2) ** 2

    norm = torch.sqrt(norm)
    if norm < max_l2_norm:
        return

    scale = max_l2_norm / (norm + eps)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a random batch from the provided dataset."""

    indices = np.random.choice(len(dataset) - context_length, size=batch_size, replace=False)
    x = torch.tensor(
        np.array([dataset[idx : idx + context_length] for idx in indices]), device=device, dtype=torch.int64
    )
    y = torch.tensor(
        np.array([dataset[idx + 1 : idx + 1 + context_length] for idx in indices]), device=device, dtype=torch.int64
    )
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """Returns the number of iterations up to the checkpoint"""
    obj = torch.load(src)

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]


def generate(
    model: torch.nn.Module,
    input: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = 0.9,
):
    logits = model(input)[...:-1:,]  # (B, V)
    probs = softmax(logits / temperature, dim=-1)

    if top_p:
        probs, _ = probs.sort(dim=-1, descending=True)
        # TODO: handle case where most likely token is >= top_p
        mask = probs.cumsum(dim=-1) <= top_p
        probs *= mask
        probs /= probs.sum(dim=-1, keepdim=True)

    vocab = torch.multinomial(probs, 1)
    return vocab.shape
