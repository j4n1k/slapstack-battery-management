import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.name = "NeuralHeuristic"
    def forward(self, x):
        return self.network(x)

    def predict(self, x, deterministic=True):
        features = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            output = self.network(features)
            action = output.argmax().item()
        return action



def train_model(train_loader, val_loader, input_dim):
    model = SimpleNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Epoch {epoch}, Accuracy: {100 * correct / total}%')

    return model