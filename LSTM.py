import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class FastLSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layers, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, z):
        b_size = z.size(0)
        h = torch.zeros(layers, b_size, hid_dim).to(z.device)
        c = torch.zeros(layers, b_size, hid_dim).to(z.device)
        out, _ = self.lstm(z, (h, c))
        return self.fc(out[:, -1, :])

def build_sequences(raw_data, seq_len):
    segments = [raw_data[i:i+seq_len] for i in range(len(raw_data) - seq_len)]
    return np.array(segments)

if __name__ == "__main__":
    seq_len = 10
    hid_dim = 32
    layers = 2
    epochs = 200
    lr = 0.001
    metrics = {}
    
    for stock in ['AMZN', 'AAPL', 'TSLA', 'MSFT']:
        print(f"\nTraining {stock}")
        data = pd.read_csv(f'CSV_Files/{stock}.csv')

        X, Y = data.iloc[:, [3] + list(range(5, 17))], data.iloc[:, 4]
        X_norm, Y_norm = MinMaxScaler(), MinMaxScaler()
        X_scaled, Y_scaled = X_norm.fit_transform(X), Y_norm.fit_transform(Y.values.reshape(-1, 1)).flatten()
        X_seq, Y_vals = build_sequences(X_scaled, seq_len), Y_scaled[seq_len:]
        X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_vals, train_size=0.8, shuffle=False)
        X_train, X_test, Y_train, Y_test = map(lambda d: torch.tensor(d, dtype=torch.float32), [X_train, X_test, Y_train, Y_test])
        Y_train, Y_test = Y_train.unsqueeze(1), Y_test.unsqueeze(1)
        in_dim = X_scaled.shape[1]

        model = FastLSTM(in_dim, hid_dim, 1, layers, 0.2)
        loss_fn = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            preds = model(X_train)
            loss = loss_fn(preds, Y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step(loss.item())

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} - MSE: {loss.item():.6f}, LR: {opt.param_groups[0]['lr']:.6f}")
                
        model.eval()
        with torch.no_grad():
            preds_test = model(X_test)
            Y_actual = Y_norm.inverse_transform(Y_test.cpu().numpy())
            actual_preds = Y_norm.inverse_transform(preds_test)
            scaled_mse = loss_fn(preds_test, Y_test).item()
            actual_mse = np.mean((actual_preds - Y_actual) ** 2)
            avg_error = np.sqrt(actual_mse)

            print(f"Test MSE (scaled): {scaled_mse:.6f}\nTest MSE (actual): {actual_mse:.6f}")
            print(f"On average, predictions are off by {avg_error:.2f} points.")

            metrics[stock] = {'scaled_mse': scaled_mse, 'actual_mse': actual_mse}
        torch.save(model.state_dict(), f"parameters/{stock}.pth")
        
    print("\nSummary:")
    print(f"{'Stock':<10} {'Scaled MSE':<15} {'Actual MSE':<15}")
    for stock, vals in metrics.items():
        print(f"{stock:<10} {vals['scaled_mse']:<15.6f} {vals['actual_mse']:<15.6f}")