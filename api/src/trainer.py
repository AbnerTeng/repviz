from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .model.ffn import OverfitFFN


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


if __name__ == "__main__":
    seed_all()
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    cov_type = fetch_covtype(data_home="covertype")
    X_train, X_test, y_train, y_test = train_test_split(
        cov_type.data, cov_type.target, test_size=0.9, random_state=42
    )
    train_dataset = SampleDataset(X_train, y_train)
    test_dataset = SampleDataset(X_test, y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    # model = FFN(X_train.shape[1], len(set(y_train)))
    # model = FFN2(X_train.shape[1], len(set(y_train))).to(device)
    # model = FFNResidual(X_train.shape[1], len(set(y_train))).to(device)
    # model = FFNKai(X_train.shape[1], len(set(y_train))).to(device)
    # model = FFNSiLU(X_train.shape[1], len(set(y_train))).to(device)
    # model = FFNLReLU(X_train.shape[1], len(set(y_train))).to(device)
    model = OverfitFFN(X_train.shape[1], len(set(y_train))).to(device)

    n_epochs = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        model.train()
        for tr_x, tr_y in train_loader:
            tr_x, tr_y = (
                torch.tensor(tr_x, dtype=torch.float32, device=device),
                torch.tensor(tr_y, dtype=torch.long, device=device),
            )
            pred_y = model(tr_x)
            loss = criterion(pred_y, tr_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))

        print(f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(train_loader)}")

    torch.save(model.state_dict(), "model/ffn_covtype_v1overfit.pth")

    with open("losses.lst", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    print("Training complete. Model saved as 'model/ffn_covtype_v1overfit.pth'.")
    print("Losses saved to 'losses.lst'.")
    print("You can now run the notebook to analyze the model.")
