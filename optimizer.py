from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]
                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Get hyperparameters
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Initialize state if this is the first step
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)  # First moment
                    state["v"] = torch.zeros_like(p.data)  # Second moment

                # Increment step count
                state["step"] += 1
                t = state["step"]

                # Get first and second moment estimates
                m = state["m"]
                v = state["v"]

                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    step_size = alpha * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = alpha

                # Update params: theta_t = theta_{t-1} - step_size * m_t / (sqrt(v_t) + eps)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)

                # weight decay (decoupled from gradient update)
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)


        return loss
