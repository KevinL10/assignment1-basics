from typing import Callable
import torch
import math


# Begin copy from CS 336 – assignment 1.
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


def example_training_loop():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer ste


# End copy


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)

                m_new = beta1 * m + (1 - beta1) * p.grad.data
                v_new = beta2 * v + (1 - beta2) * p.grad.data**2

                m, v = m_new, v_new
                lr_t = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * m / (v**0.5 + group["eps"])
                p.data -= group["lr"] * group["weight_decay"] * p.data

                group["current_lr"] = lr_t

                state["t"] = t + 1
                state["m"] = m_new
                state["v"] = v_new

        return loss


if __name__ == "__main__":
    example_training_loop()
