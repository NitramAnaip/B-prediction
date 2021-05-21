import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = 0.1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.d = nn.Dropout(p=0.1)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = torch.relu_(self.fc(out[:, -1, :]) )
        out = torch.sigmoid(self.fc2(out))
        out = self.d(out)

        return out



class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout = 0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.d = nn.Dropout(p=0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out



import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.fc0 = nn.Linear(180,180)
        self.fc1 = nn.Linear(180, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.2)

    def _conv_layer_set(self, in_c, out_c, maxpool_size):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_c, out_c, kernel_size=(1, maxpool_size, maxpool_size), padding=0
            ),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 3, 3)),
        )
        return conv_layer

    def forward(self, input_data):
        # Set 1
        
        out = input_data.view(input_data.shape[0], -1)
        out = self.fc0(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.fc3(out)
        return self.sigmoid(out)
