import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


class LowLevelModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class HighLevelModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class AdaptiveHaltHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.fc1(state))
        y = torch.sigmoid(self.fc2(y))  # scalar between zero and one
        return y.squeeze(-1)


class HierarchicalReasoningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_heads: int = 12,
        num_low_blocks: int = 2,
        num_high_blocks: int = 2,
        micro_steps: int = 4,
        cycles: int = 8,
        use_act: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.low_module = LowLevelModule(d_model, n_heads, num_low_blocks)
        self.high_module = HighLevelModule(d_model, n_heads, num_high_blocks)
        self.norm_out = nn.LayerNorm(d_model)
        self.head_out = nn.Linear(d_model, vocab_size, bias=False)

        self.micro_steps = micro_steps
        self.cycles = cycles
        self.use_act = use_act
        if use_act:
            self.halt_head = AdaptiveHaltHead(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        max_cycles: Optional[int] = None,
        teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_cycles = max_cycles or self.cycles
        batch_size, seq_len = input_ids.size()
        x = self.embed(input_ids)
        high_state = torch.zeros_like(x)
        used_cycles = torch.zeros(batch_size, device=input_ids.device)

        for cycle in range(max_cycles):
            low_state = high_state
            for _ in range(self.micro_steps):
                low_state = self.low_module(low_state)

            high_state = self.high_module(low_state.detach())  # detach to keep memory constant

            if self.use_act and not teacher_forcing:
                prob_halt = self.halt_head(high_state[:, -1])  # use last token state
                should_halt = (prob_halt > 0.5).float()
                used_cycles += 1.0
                if should_halt.all():
                    break
            else:
                used_cycles += 1.0

        logits = self.head_out(self.norm_out(high_state))
        return logits, used_cycles


if __name__ == "__main__":
    model = HierarchicalReasoningModel()
    dummy = torch.randint(0, 32000, (2, 16))
    out, steps = model(dummy)
    print(out.shape, steps)
