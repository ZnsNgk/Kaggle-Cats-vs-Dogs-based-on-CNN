import torch
from torch import nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2,bias=True)
        self.pool1 = nn.MaxPool2d(2,2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=True)
        self.pool2 = nn.MaxPool2d(2,2)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*28*28,16)
        self.fc2 = nn.Linear(16,2)
        self.out = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

if __name__=='__main__':
    model = LeNet()
    print(model)
    input = torch.randn(1, 1, 112, 112)
    out = model(input)
    print(out)
        