# Convex architecture neural network
import torch
import torch.nn as nn
class NetConvex(nn.Module):
    def __init__(self):
        super(NetConvex, self).__init__()
        self.fc1 = nn.Linear(2,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,1)
        self.skip1 = nn.Linear(2,256)
        self.skip2 = nn.Linear(2,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) + self.skip1(y)
        x = self.relu(x)
        x = self.fc3(x) + self.skip2(y)
        x = self.sigmoid(x)
        
        return x
