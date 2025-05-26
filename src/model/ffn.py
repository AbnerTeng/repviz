import torch
from torch import nn


class FFN(nn.Module):
    def __init__(self, n_feats: int, n_classes: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.Dropout(0.1),
            nn.Linear(n_feats, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
