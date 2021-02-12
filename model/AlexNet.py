import torch
import torch.nn as nn
import torch.functional as F
 
class AlexNet(nn.Module):   #定义网络
    def  __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0) #input_size = 227*227*1, output_size = 55*55*96
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(num_features=96)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)   #output_size = 27*27*96
        self.conv2 = nn.Sequential(   #input_size = 27*27*96
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2),    #output_size = 13*13*256
        )
        self.conv3 = nn.Sequential(   #input_size = 13*13*256
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),    #output_size = 13*13*384
        )
        self.conv4 = nn.Sequential(   #input_size = 13*13*384
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),    #output_size = 13*13*384
        )
        self.conv5 = nn.Sequential(   #input_size = 13*13*384
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),    #output_size = 6*6*256
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=6*6*256,out_features=4096),
            nn.ReLU(),
        )
        self.drop1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.drop2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(4096,2)
        self.out = nn.Softmax(dim=1)
 
    def forward(self, x):   #正向传播过程
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x

if __name__=='__main__':
    model = AlexNet()
    print(model)
    input = torch.randn(1, 3, 227, 227)
    out = model(input)
    print(out)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    