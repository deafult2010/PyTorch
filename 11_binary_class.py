import torch
import torch.nn as nn


class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Layer(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Layer(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred


model = NeuralNet2(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
