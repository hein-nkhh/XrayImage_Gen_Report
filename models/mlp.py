import torch
import torch.nn as nn
from config import MLP_HIDDEN_DIM

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, MLP_HIDDEN_DIM)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(MLP_HIDDEN_DIM, MLP_HIDDEN_DIM)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(MLP_HIDDEN_DIM, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.norm(x)
