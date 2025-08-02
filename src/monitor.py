import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hooks import HookManager


class TrainingMonitor:
    def __init__(self, model, track_all=True):
        self.model = model
        self.hook_mgr = HookManager(track_all=track_all)
        self.hook_mgr.register_hooks(self.model)

    def train(self, features, labels, batch_size=64, epochs=10):
        dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for i, (x_batch, y_batch) in enumerate(tqdm(loader)):
                self.model.train()
                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()

    def get_all_data(self):
        return {
            "activations": self.hook_mgr.get_activations(),
            "inputs": self.hook_mgr.get_inputs(),
            "gradients": self.hook_mgr.get_gradients(),
            "weights": self.hook_mgr.get_weights(),
        }
