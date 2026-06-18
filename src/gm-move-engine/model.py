import torch.nn as nn 
import torch

class ChessCNN(nn.Module):
    def __init__(self,num_moves):
        super().__init__()

        self.conv1 = nn.Conv2d(18,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,128,3,padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128,128,3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*8*8,512)

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512,num_moves)


    def forward(self,x):

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        x = torch.flatten(x,start_dim=1)

        x = torch.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x
