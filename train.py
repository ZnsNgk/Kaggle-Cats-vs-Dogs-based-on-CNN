import random, os, glob
import torch
import math
import numpy
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model import AlexNet
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

CLASSES = {0: "cat", 1: "dog"}
train_folder = './train'    #训练集文件夹
val_folder = './validation'    #验证集文件夹
data_transform = transforms.Compose([
        transforms.Resize([227,227]),  # 把尺寸变换为[227,227]
        transforms.ToTensor(),
# 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloatTensor /255.操作
    ])
#torch.manual_seed(1)    #设置随机化种子，确保网络可复现
batch_size = 8   #批次大小
lr = 0.0001   #初始学习率
Epoch = 20   #迭代轮数

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, img_folder, transform=None):
        self.transform = transform

        dog_dir = os.path.join(img_folder, "dog")
        cat_dir = os.path.join(img_folder, "cat")
        imgsLib = []
        imgsLib.extend(glob.glob(os.path.join(dog_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(cat_dir, "*.jpg")))
        random.shuffle(imgsLib)  # 打乱数据集
        self.imgsLib = imgsLib
    def __getitem__(self, index):
        img_path = self.imgsLib[index]
        if 'dog' in img_path: #狗的label设为1，猫的设为0
            label = 1
        else:
            label = 0
        img = Image.open(img_path).convert("RGB")     #读取图片
        img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgsLib)

train_data = MyDataSet(train_folder, transform=data_transform)
val_data = MyDataSet(val_folder, transform=data_transform)
train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
val_dataLoader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

model = AlexNet.AlexNet()   #导入模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #自动检测是否支持CUDA
#device = 'cpu'   #强制使用cpu
model.to(device)
loss_function = nn.CrossEntropyLoss()   #定义损失函数
optim = optim.Adam(model.parameters(),lr)   #定义优化器
scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=60,gamma=0.5)     #学习率衰减，每迭代60次学习率变为原来的一半
train_loss_list = []
val_loss_list = []
train_accuracy_list = []
val_accuracy_list = []

def fit(epoch,model,data_loader,phase,volatile=False):   #自定义的训练和验证的函数
    if phase == 'training':
        torch.set_grad_enabled(True)
        model.train()
    if phase == 'validation':
        torch.set_grad_enabled(False)
        model.eval()
        volatile = True
    running_loss = 0.0
    accuracy = 0.0
    running_acc = 0.0
    total = 0   #记录i取到的最大值(即每个epoch的step数)
    for i , data in enumerate(data_loader):
        img, lab = data
        if phase == 'training':
            optim.zero_grad()
        out = model(img.to(device, torch.float))
        predict = out.argmax(dim=1)
        accuracy += (predict.data.cpu() == lab.data).sum()
        loss = loss_function(out, lab.to(device, torch.long))
        running_loss += loss
        total = i
        if phase == 'training':
            loss.backward()
            optim.step()
            scheduler.step()
    running_loss = running_loss / (total + 1)   #计算该轮次中loss平均值
    running_acc = accuracy / ((total + 1) * batch_size)    #计算该轮次的平均准确率
    return running_loss, running_acc

def train():   #训练
    for epoch in range(Epoch):
        train_loss, train_acc = fit(epoch,model,train_dataLoader,'training')
        val_loss, val_acc = fit(epoch,model,val_dataLoader,"validation")
        train_acc = train_acc * 100
        val_acc = val_acc * 100
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)
        print('Epoch: '+str(epoch+1)+' | train loss:'+str(round(float(train_loss),6))+'| train accuracy:'+str(round(float(train_acc),2))+'% | val loss:'+str(round(float(val_loss),6))+' | val accuracy:'+str(round(float(val_acc),2))+'%')
    print('Finished Training!')
    PATH = './model_parameter.pth'
    torch.save(model.state_dict(), PATH)     #只保存网络里的参数
    #PATH = './model.pkl'
    #torch.save(model, PATH)   #保存整个网络和参数

def drew_result():#画图部分
    plt.figure(figsize = (14,7))
    model_name = model.__class__.__name__
    plt.suptitle(model_name + ' Cats vs Dogs Result')
    plt.subplot(121)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(Epoch),train_loss_list,label="train")
    plt.plot(range(Epoch),val_loss_list,label="val")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    plt.subplot(122)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.ylim(ymax=100,ymin=0)
    plt.plot(range(Epoch),train_accuracy_list,label="train")
    plt.plot(range(Epoch),val_accuracy_list,label="val")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    plt.savefig("train result.png",dpi=600)     #将损失曲线保存成图片
    plt.show()

 
if __name__ == '__main__':
    train()
    drew_result()
