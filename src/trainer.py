from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .mapper import mapper
from .model.ffn import FFN, FFN2, FFNResidual, FFNKai, FFNSiLU, FFNLReLU, OverfitFFN


class SampleDataset(Dataset):
    def __init__(self, features: np.ndarray, label: np.ndarray) -> None:
        self.features = features
        self.label = label
        self._scaler()

    def _scaler(self) -> None:
        mm = MinMaxScaler()
        self.features = mm.fit_transform(self.features)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.features[idx], self.label[idx] - 1


def seed_all(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="ffn",
        choices=["ffn", "ffn2", "ffn_residual", "ffn_kai", "ffn_silu", "ffn_lrelu", "overfit_ffn"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    seed_all()
    args = get_args()
    wandb.init(
        project="repviz-mlp",
        name=f"{args.model_type}_expr_tr005"
    )

    device = torch.device("cuda")
    # device = (
    #     torch.device("mps")
    #     if torch.backends.mps.is_available()
    #     else torch.device("cpu")
    # )
    cov_type = fetch_covtype(data_home="covertype")
    X_train, X_valid, y_train, y_valid = train_test_split(
        cov_type.data, cov_type.target, test_size=0.9, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid, y_valid, test_size=0.9, random_state=42
    ) 
    train_dataset = SampleDataset(X_train, y_train)
    valid_dataset = SampleDataset(X_valid, y_valid) 
    test_dataset = SampleDataset(X_test, y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=len(y_valid), shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    
    model = mapper[args.model_type](X_train.shape[1], len(set(y_train))).to(device)
    wandb.watch(model, log="all", log_freq=10)
    n_epochs = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        model.train()
        for tr_x, tr_y in train_loader:
            tr_x, tr_y = (
                torch.tensor(tr_x, dtype=torch.float32, device=device),
                torch.tensor(tr_y, dtype=torch.long, device=device),
            )
            pred_y_logit = model(tr_x)
            pred_y_clf = torch.argmax(pred_y_logit, dim=-1)
            loss = criterion(pred_y_logit, tr_y)
            accuracy = (pred_y_clf == tr_y).float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy

        if epoch % 10 == 0:
            print(f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(train_loader)} | Accuracy: {epoch_accuracy / len(train_loader)}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / len(train_loader),
                "accuracy": epoch_accuracy / len(train_loader),
            }
        )

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_accuracy = 0.0
            for val_x, val_y in tqdm(valid_loader):
                val_x, val_y = (
                    torch.tensor(val_x, dtype=torch.float32, device=device),
                    torch.tensor(val_y, dtype=torch.long, device=device),
                )
                pred_y_logit = model(val_x)
                loss = criterion(pred_y_logit, val_y)
                accuracy = (torch.argmax(pred_y_logit, dim=-1) == val_y).float().mean().item()
                valid_loss += loss.item()
                valid_accuracy += accuracy

            print(f"Validation Loss: {valid_loss / len(valid_loader)} | Accuracy: {valid_accuracy / len(valid_loader)}")
            wandb.log(
                {
                    "valid_loss": valid_loss / len(valid_loader),
                    "valid_accuracy": valid_accuracy / len(valid_loader),
                }
            )

    torch.save(model.state_dict(), "model/ffn_covtype_v1overfit.pth")

    print("Training complete. Model saved as 'model/ffn_covtype_v1overfit.pth'.")
    print("Losses saved to 'losses.lst'.")
    print("You can now run the notebook to analyze the model.")
    wandb.finish()
