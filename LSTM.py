import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  
        
        out = self.fc(out[:, -1, :])   
        return out

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    
    return np.array(sequences)

if __name__ == "__main__":
    seq_length = 10
    hidden_size = 32
    num_layers = 2
    num_epochs = 200
    learning_rate = 0.001
    
    results = {}
    for name in ['AMZN', 'AAPL', 'TSLA', 'MSFT']:
        print(f"\n{'='*50}")
        print(f"Training for {name}")
        print(f"{'='*50}")
        
        df = pd.read_csv(f'CSV_Files/{name}.csv')

        x = df.iloc[:, [3] + list(range(5, 17))]
        y = df.iloc[:, 4]
        
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        x_scaled = x_scaler.fit_transform(x)
        y_reshaped = y.values.reshape(-1, 1)
        y_scaled = y_scaler.fit_transform(y_reshaped).flatten()
        
        x_sequences = create_sequences(x_scaled, seq_length)
        y_values = y_scaled[seq_length:]
        
        # Split into training and testing sets - no shuffling to preserve time order
        x_train, x_test, y_train, y_test = train_test_split(
            x_sequences, y_values, train_size=0.8, shuffle=False
        )
        
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        
        input_size = x_scaled.shape[1]
        model = LSTM(input_size=input_size, hidden_size=hidden_size, 
                   output_size=1, num_layers=num_layers, dropout=0.2)
        
        
        
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop - removed batching, process entire dataset at once
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(x_train_tensor)
            loss = loss_function(outputs, y_train_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(loss.item())
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train MSE: {loss.item():.6f}, LR: {current_lr:.6f}")
        
        model.eval()
        with torch.no_grad():
            predictions = model(x_test_tensor)
            y_test_actual = y_scaler.inverse_transform(y_test_tensor.cpu().numpy())
            test_loss = loss_function(predictions, y_test_tensor)
            predictions_actual = y_scaler.inverse_transform(predictions)

            actual_mse = np.mean((predictions_actual - y_test_actual) ** 2)

            print(f"Test MSE (scaled): {test_loss.item():.6f}")
            print(f"Test MSE (actual stock prices): {actual_mse:.6f}")
            
            results[name] = {
                'scaled_mse': test_loss.item(),
                'actual_mse': actual_mse
            }
        
        torch.save(model.state_dict(), f"parameters/{name}.pth")
    
    print("\nResults Summary:")
    print(f"{'Stock':<10} {'Scaled MSE':<15} {'Actual MSE':<15}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<10} {metrics['scaled_mse']:<15.6f} {metrics['actual_mse']:<15.6f}")