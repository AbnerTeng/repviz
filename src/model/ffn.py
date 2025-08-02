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


class FFNKai(nn.Module):
    def __init__(self, n_feats: int, n_classes: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.Dropout(0.1),
            nn.Linear(n_feats, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class FFNSiLU(nn.Module):
    def __init__(self, n_feats: int, n_classes: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.Dropout(0.1),
            nn.Linear(n_feats, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class FFNLReLU(nn.Module):
    def __init__(self, n_feats: int, n_classes: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(n_feats),
            nn.Dropout(0.1),
            nn.Linear(n_feats, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class FFN2(nn.Module):
    def __init__(self, n_feats: int, n_classes: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Dropout(0),
            nn.Linear(n_feats, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class FFNResidual(nn.Module):
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
        )
        self.residual = nn.Linear(n_feats, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.ffn(x) + self.residual(x))


class OverfitFFN(nn.Module):
    def __init__(self, n_feats: int, n_classes: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_feats, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
