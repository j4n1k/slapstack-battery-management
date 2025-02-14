import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_storage_strategy_dataframe(data_root):
    n_zones = 3
    try:
        strategy_name = data_root.split('/')[3]
        # n_zones = int(strategy_name[-1]) if strategy_name[-1].isdigit() else 3
    except:
        strategy_name = "DQN"
    if not os.path.exists(data_root):
        print(f"did not find path {data_root}; skipping...")
        return
    dfs = []
    csv_f_names = os.listdir(data_root)
    # pbar = tqdm(total=len(csv_f_names))
    print(f'Loading result files into dataframes for the '
          f'{strategy_name} simulation run...')
    for f_name in csv_f_names:
        if os.path.isdir(f'{data_root}/{f_name}') or f_name == '.DS_Store':
            #print(f'{data_root}/{f_name}')
            continue
        df_result_part = pd.read_csv(f'{data_root}/{f_name}', index_col=0)
        n_rows = df_result_part.shape[0]
        df_result_part['strategy_name'] = [strategy_name] * n_rows
        df_result_part['n_zones'] = [n_zones] * n_rows
        dfs.append(df_result_part)
        # print(strategy_name, n_zones, order_set_nr)
        # pbar.update(1)
    strategy_df = pd.concat(dfs).reset_index(drop=True)
    strategy_df.name = strategy_name
    return strategy_df

# Data Preparation Functions
def get_incremental_values(df):
    """Calculate incremental values for cumulative metrics"""
    incremental_features = ['total_distance', 'total_shift_distance', 'n_finished_orders',
                            'n_pallet_shifts', 'n_steps', 'n_decision_steps', 'n_charging_events']

    for col in incremental_features:
        df[f'{col}_per_order'] = df[col].diff()
        df[f'{col}_per_order'].iloc[0] = df[col].iloc[0]

    return df


def prepare_time_features(df):
    """Prepare time features while maintaining original time information"""
    # Keep original timestamp for positional encoding
    seconds_in_day = 24 * 3600
    df['timestamp'] = df['kpi__makespan']

    # Calculate relative time features
    # df['time_since_start'] = df['timestamp'] - df['timestamp'].iloc[0]
    df['time_between_orders'] = df['timestamp'].diff()
    df['time_between_orders'].fillna(0, inplace=True)
    df['day'] = (df['kpi__makespan'] // seconds_in_day) % 7
    df['hour_sin'] = np.sin(2 * np.pi * ((df["kpi__makespan"] % seconds_in_day) / 3600) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * ((df["kpi__makespan"] % seconds_in_day) / 3600) / 24)

    return df


def prepare_df(df):
    """Main data preparation function"""
    # Prepare timestamps first
    df = prepare_time_features(df)

    # Get incremental values
    df = get_incremental_values(df)

    # Define feature groups
    time_cols = ['day', 'hour_sin', 'hour_cos']
    distance_cols = ['total_distance_per_order', 'total_shift_distance_per_order',
                     'average_distance', 'travel_time_retrieval_ave', 'distance_retrieval_ave']
    count_cols = ['n_queued_retrieval_orders', 'n_queued_delivery_orders',
                  'n_pallet_shifts_per_order', 'n_charging_events_per_order']
    agv_cols = ['n_free_agvs', 'avg_battery_level', 'n_agv_depleted',
                'n_agv_not_depleted', 'n_queued_charging_events']
    normalized_cols = ['fill_level', 'entropy', 'utilization_time'] #'utilization_time'

    # Combine all feature columns
    feature_cols = time_cols + count_cols + agv_cols + normalized_cols
    target_col = 'kpi__average_service_time'

    # Remove any constant columns
    constant_cols = [col for col in feature_cols if df[col].nunique() == 1]
    feature_cols = [col for col in feature_cols if col not in constant_cols]

    # Scale features
    scaler = StandardScaler()
    X = df[feature_cols].values
    y = df[target_col].values

    return X, y, feature_cols, scaler


# Model Components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, timestamps):
        # Scale timestamps using log1p
        scaled_time = torch.log1p(timestamps)

        batch_size, seq_len = x.shape[0], x.shape[1]
        position = scaled_time.unsqueeze(-1)

        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             (-math.log(10000.0) / self.d_model)).to(x.device)

        pe = torch.zeros(batch_size, seq_len, self.d_model).to(x.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return x + pe


class ServiceTimeTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, src, timestamps):
        x = self.input_projection(src)
        x = self.pos_encoder(x, timestamps)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take last sequence element
        output = self.decoder(x)
        return output


# Training and Evaluation Functions
def train_model(model, train_loader, val_loader, target_scaler, epochs=100, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, timestamps, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X, timestamps).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, timestamps, batch_y in val_loader:
                output = model(batch_X, timestamps).squeeze()
                val_loss += criterion(output, batch_y).item()

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    # Load best model
    model.load_state_dict(best_model)
    return model


def evaluate_model(model, val_loader, target_scaler):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, timestamps, batch_y in val_loader:
            output = model(batch_X, timestamps).squeeze()
            # Transform back to original scale
            pred = target_scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1))
            actual = target_scaler.inverse_transform(batch_y.cpu().numpy().reshape(-1, 1))
            predictions.extend(pred)
            actuals.extend(actual)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Service Time')
    plt.ylabel('Predicted Service Time')
    plt.title('Predicted vs Actual Service Times')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nModel Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return predictions, actuals


def train_and_evaluate_xgboost(X, y, feature_cols):
    # Split data (80/20)
    train_size = int(0.9 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Initialize and train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    # Make predictions
    predictions = model.predict(X_val)

    # Calculate metrics
    mse = np.mean((predictions - y_val) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_val))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, predictions, alpha=0.5)
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Service Time')
    plt.ylabel('Predicted Service Time')
    plt.title('Predicted vs Actual Service Times')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nModel Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Feature importance
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_imp.head(10))

    return model, predictions


# Main execution
if __name__ == "__main__":
    # Parameters
    seq_length = 24
    batch_size = 32
    epochs = 000
    learning_rate = 0.001

    # Load and prepare data
    path = f'../experiments/result_data_wepa/full_run'
    df = load_storage_strategy_dataframe(path)
    print(df.shape)
    # df = pd.read_csv("./data/raw/pt_1_COL_opportunity_thopportunity_1001.csv")
    # df_scaled, feature_cols, target_col, scaler, target_scaler = prepare_df(df)


    # Create sequences
    # X, timestamps, y = [], [], []
    # for i in range(len(df_scaled) - seq_length):
    #     X.append(df_scaled[feature_cols].iloc[i:i + seq_length].values)
    #     timestamps.append(df_scaled['timestamp'].iloc[i:i + seq_length].values)
    #     y.append(df_scaled[target_col].iloc[i + seq_length])

    # X = np.array(X)
    # timestamps = np.array(timestamps)
    # y = np.array(y)

    # Split data
    # train_size = int(0.7 * len(df_scaled))
    # X_train, X_val = X[:train_size], X[train_size:]
    # timestamps_train = timestamps[:train_size]
    # timestamps_val = timestamps[train_size:]
    # y_train, y_val = y[:train_size], y[train_size:]
    X, y, feature_cols, scaler = prepare_df(df)
    model, predictions = train_and_evaluate_xgboost(X, y, feature_cols)

    # Create datasets and loaders
    # train_dataset = TensorDataset(
    #     torch.FloatTensor(X_train).to(device),
    #     torch.FloatTensor(timestamps_train).to(device),
    #     torch.FloatTensor(y_train).to(device)
    # )
    # val_dataset = TensorDataset(
    #     torch.FloatTensor(X_val).to(device),
    #     torch.FloatTensor(timestamps_val).to(device),
    #     torch.FloatTensor(y_val).to(device)
    # )
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and train model
    # model = ServiceTimeTransformer(
    #     input_dim=len(feature_cols),
    #     d_model=128,
    #     nhead=8,
    #     num_layers=4,
    #     dropout=0.1
    # )
    #
    # model = train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     target_scaler,
    #     epochs=epochs,
    #     learning_rate=learning_rate
    # )
    #
    # # Evaluate model
    # predictions, actuals = evaluate_model(model, val_loader, target_scaler)