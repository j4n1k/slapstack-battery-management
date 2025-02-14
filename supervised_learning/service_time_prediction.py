import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class SequentialChargingDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, seq_length=10):
        """
        Args:
            data: pandas DataFrame
            feature_cols: list of feature column names
            target_col: target column name
            seq_length: number of time steps in sequence
        """
        self.features = torch.FloatTensor(data[feature_cols].values)
        self.targets = torch.FloatTensor(data[target_col].values)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        return (self.features[idx:idx + self.seq_length],
                self.targets[idx + self.seq_length])


class ChargingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :])
        return self.relu(output)


def prepare_df(path_go_charging: str):
    data_go_charging = pd.read_csv(path_go_charging, index_col=0)
    data_go_charging["service_time"] = data_go_charging["service_time"] #/ 1000
    return data_go_charging


def get_sequential_loader(path_go_charging: str, seq_length=10, batch_size=32):
    data_go_charging = prepare_df(
        path_go_charging)

    target_col = "service_time"
    feature_cols = data_go_charging.columns[3:]  # Assuming first 3 columns are metadata

    # Create sequential dataset
    dataset = SequentialChargingDataset(
        data_go_charging, feature_cols, target_col, seq_length)

    # Create dataloader without shuffling to maintain sequence order
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader, dataset, feature_cols


def train_model(model, train_loader, val_loader, epochs=1000,
                learning_rate=0.001, early_stopping_patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min',
    #                                           factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for seq_X, seq_y in train_loader:
            optimizer.zero_grad()
            outputs = model(seq_X).squeeze()
            loss = criterion(outputs, seq_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq_X, seq_y in val_loader:
                    outputs = model(seq_X).squeeze()
                    val_loss += criterion(outputs, seq_y).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

            # Learning rate scheduling
            # scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    model.load_state_dict(best_model)
                    return model

    model.load_state_dict(best_model)
    return model


def evaluate_model(model, val_loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for seq_X, seq_y in val_loader:
            outputs = model(seq_X).squeeze()
            predictions.extend(outputs.numpy())
            actuals.extend(seq_y.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)

    print("\nError Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)],
             [min(actuals), max(actuals)], 'r--',
             label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(predictions - actuals, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, actuals


# Main execution
if __name__ == "__main__":
    # Parameters
    seq_length = 10
    batch_size = 32
    hidden_dim = 64
    epochs = 5000
    learning_rate = 0.001

    # Get data loaders
    train_loader, train_dataset, feature_cols = get_sequential_loader(
        "data/raw/week_1/data.csv",
        seq_length=seq_length,
        batch_size=batch_size
    )

    val_loader, val_dataset, _ = get_sequential_loader(
        "data/raw/week_2/data.csv",
        seq_length=seq_length,
        batch_size=batch_size
    )

    print(feature_cols)
    # Initialize and train model
    model = ChargingLSTM(input_dim=len(feature_cols), hidden_dim=hidden_dim)
    model = train_model(model, train_loader, val_loader,
                        epochs=epochs, learning_rate=learning_rate)

    # Evaluate model
    predictions, actuals = evaluate_model(model, val_loader)