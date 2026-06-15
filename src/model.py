import torch
import torch.nn as nn

class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(18,32,3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(4096,128)
        self.fc2 = nn.Linear(128,1)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)
        
        x = torch.flatten(x,start_dim=1)
        
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        
        x = torch.tanh(x)
        
        return x
