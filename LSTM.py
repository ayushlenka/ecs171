import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))  
        
        out = self.fc(out[:, -1, :])   
        return out

if __name__ == "__main__":
    df = pd.read_csv('CSV_Files/AAPL.csv')

    x = df.iloc[:, [3] + list(range(5, 17))]
    y = df.iloc[:, 4]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    x_train_tensor = x_train_tensor.unsqueeze(1) 
    x_test_tensor  = x_test_tensor.unsqueeze(1)   

    model = LSTM(input_size=13, hidden_size=10, output_size=1, num_layers=2)

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_train_tensor)
        loss = loss_function(y_pred, y_train_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1} - Train MSE: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test_tensor)
        test_mse = loss_function(y_pred_test, y_test_tensor)
        print(f"Test MSE: {test_mse.item():.6f}")


