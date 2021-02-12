import os
import torch
import math
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import AlexNet
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

CLASSES = {0: "cat", 1: "dog"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'  #强制使用CPU
testfolder = './test'
data_transform = transforms.Compose([
        transforms.Resize([227,227]), 
        transforms.ToTensor(),
    ])

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, img_folder, transform=None):
        self.transform = transform
        self.imageFolder = img_folder
        self.imgs = os.listdir(img_folder)
    def __getitem__(self, index):
        img_name = self.imgs[index]
        img_path = os.path.join(self.imageFolder, img_name)
        img = Image.open(img_path).convert("RGB")
        img_ori = img   #获取原图以作备用
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img, img_ori, img_name
    def __len__(self):
        return len(self.imgs)

def test(imageFolder):      #测试部分
    is_paramatter = False   #置False为导入整个模型，置True为导入参数文件
    if(is_paramatter):
        net = AlexNet()
        model = torch.load('./model_parameter.pth',map_location=torch.device(device))#模型参数文件
        net.load_state_dict(model)
    else :
        net = torch.load('./model.pkl', map_location=torch.device(device))
    net = net.to(device)
    torch.set_grad_enabled(False)
    torch.no_grad()
    net.eval()
    data_num = MyDataSet(imageFolder).__len__()
    for i in range(data_num):
        img, ori, name = MyDataSet(imageFolder,data_transform).__getitem__(i)
        out = net(img.to(device, torch.float))
        predict = out.argmax(dim=1)     #预测的label
        probability = out[:,predict]      #该label的概率
        s = 'Predict result: This is a '
        if predict == 0:
            s += 'CAT'
        else:
            s += 'DOG'
        s += ' with the probability of '
        s += str(round(float(probability),4))
        plt.title(s)
        plt.imshow(ori)
        plt.savefig("./result/"+name.replace('.jpg','')+".png",dpi=300)    #将结果保存在result文件夹内
        plt.show()      #显示图片
        print(name+' Success!')

if __name__ == '__main__':
    test(testfolder)
