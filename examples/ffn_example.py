from typing import Tuple

import numpy as np
import torch
from alive_progress import alive_it
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.hooks import HookManager
from src.utils import list_child_modules


class SampleDataset(Dataset):
    def __init__(self, features: np.ndarray, label: np.ndarray) -> None:
        super().__init__()
        self.features = features
        self.label = label
        self._scaler()

    def _scaler(self) -> None:
        mm = MinMaxScaler()
        self.features = mm.fit_transform(self.features)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return (self.features[idx, :], self.label[idx] - 1)


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
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)



if __name__ == "__main__":
    cov_type = fetch_covtype(data_home="covertype")
    X_train, X_test, y_train, y_test = train_test_split(
        cov_type.data, cov_type.target, test_size=0.2, random_state=42
    )
    train_dataset = SampleDataset(X_train, y_train)
    test_dataset = SampleDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("training...")

    model = FFN(X_train.shape[1], len(set(y_train)))
    list_child_modules(model)

    n_epochs = 1
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model = FFN(X_train.shape[1], len(set(y_train)))

    for epoch in alive_it(range(n_epochs), bar="fish"):
        epoch_loss = 0
        model.train()
        for tr_x, tr_y in train_loader:
            tr_x, tr_y = (
                torch.tensor(tr_x, dtype=torch.float32),
                torch.tensor(tr_y, dtype=torch.long),
            )
            pred_tr_y = model(tr_x)
            loss = criterion(pred_tr_y, tr_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(train_loader)}")

    print("testing...")
    model.eval()
    hook_mgr = HookManager()
    hook_mgr.register_hooks(model, partial_match="LayerNorm")
    with torch.no_grad():
        preds = []

        for ts_x, ts_y in test_loader:
            ts_x = torch.tensor(ts_x, dtype=torch.float32)
            ts_y = torch.tensor(ts_y, dtype=torch.long)
            pred_y_test = model(ts_x)
            preds.append(pred_y_test)

        print(
            f"Accuracy: {np.sum(np.argmax(np.array(preds), dim=1) == y_test).item() / len(y_test)}"
        )

    activations = hook_mgr.get_activations()