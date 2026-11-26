"""AdamW for BF16 parameters with stochastic rounding.

Why: BF16 has an 8-bit mantissa. For a typical weight (~0.03) neighbouring BF16 values are ~1e-4 apart, so an
AdamW step of ~1e-5 rounds to nothing under round-to-nearest and training silently stalls. Stochastic
rounding makes the update unbiased in expectation: weights genuinely move, one random rounding at a time.

Moments are kept in FP32 (they are not 'master weights'; nothing is FP8). The parameter update is computed in
FP32 and written back to the BF16 parameter. Used identically for both arms.
"""
import torch


def stochastic_round_to_bf16(x32: torch.Tensor) -> torch.Tensor:
    """Round an FP32 tensor to BF16, rounding up with probability proportional to the discarded fraction."""
    bits = x32.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=x32.device)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class BF16SRAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, stochastic_round=True):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))
        self.stochastic_round = stochastic_round

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p, dtype=torch.float32)
                    st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                g = p.grad.float()
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
                denom = (st["v"] / bc2).sqrt_().add_(eps)
                p32 = p.float()
                if wd:
                    p32.mul_(1 - lr * wd)
                p32.addcdiv_(st["m"], denom, value=-lr / bc1)
                if p.dtype == torch.bfloat16 and self.stochastic_round:
                    p.copy_(stochastic_round_to_bf16(p32))
                else:
                    p.copy_(p32)
