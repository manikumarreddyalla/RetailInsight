import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ---------- LSTM MODEL ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ---------- TRAIN FUNCTION ----------
def train_retail_model(clean_csv_path):
    df = pd.read_csv(clean_csv_path)

    # Basic check
    if "sales" not in df.columns:
        raise ValueError("Dataset must contain 'sales' column.")

    values = df["sales"].values.astype(float)

    # ----- Create sequences for LSTM -----
    seq_len = 14
    X_seq, y_seq = [], []
    for i in range(len(values) - seq_len):
        X_seq.append(values[i:i+seq_len])
        y_seq.append(values[i+seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # ---------- Train XGBoost ----------
    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5
    )
    xgb.fit(X_train, y_train)

    # ---------- Train LSTM ----------
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    for epoch in range(30):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    # Save both models
    joblib.dump(xgb, "xgb_model.pkl")
    torch.save(model.state_dict(), "lstm_model.pt")

    return {
        "status": "success",
        "message": "Models trained and saved.",
        "training_loss": float(loss.item())
    }

