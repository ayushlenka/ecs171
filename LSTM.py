import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, input):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        output, _ = self.lstm(input, (h0, c0))
        output = self.output(output[:, -1])

        return output
    

if __name__ == "__main__":
    model = LSTM(13, 10, 1, 2, 10)

    df = pd.read_csv('CSV_Files/AAPL.csv')

    x = df.iloc[:, [3] + list(range(5, 17))]
    y = df.iloc[:, 4]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.bool)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.bool)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.bool)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.bool)

    print(x_train_tensor.shape)

    lossFunction = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03)

    num_epochs = 500

    for epoch in range(num_epochs):
        model.train()

        output = model(x_train_tensor)
        loss = lossFunction(output, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                MSE = F.mse_loss(output, y_train_tensor)
                print(f'Epoch {epoch + 1}: MSE = {MSE}')

    model.eval()
    with torch.no_grad():
        y_pred = model(x_train_tensor)
        MSE = F.mse_loss(y_pred, y_test_tensor)
        print(f"Train MSE: {MSE}")