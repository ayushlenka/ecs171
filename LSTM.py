import torch.nn as nn
import torch
import os

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
    model = LSTM(2, 10, 1, 2, 10)

