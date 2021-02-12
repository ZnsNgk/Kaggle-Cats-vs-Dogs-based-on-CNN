# Kaggle-Cats-vs-Dogs-based-on-CNN

--------------------------------------

CNN教学示例代码，基于pytorch，使用Kaggle的猫狗大战数据集，AlexNet网络进行分类

--------------------------------------

文件结构：

model文件夹：存放模型文件，本示例存放了LeNet和AlexNet的代码，默认使用AlexNet

result文件夹：存放测试结果，其结果被保存为图片，标注属于猫或狗的哪一类，并给出属于该类的概率

test文件夹：存放测试集

train文件夹：存放训练集，其中包含cat和dog两个文件夹，两个文件夹内分别存放相应标签的图片（本示例仅从数据集中选取了100张图片作为训练集，完整数据集可以在https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset 下载，并需要手动划分训练集和验证集）

validation文件夹：存放验证集，其中包含cat和dog两个文件夹，两个文件夹内分别存放相应标签的图片（本示例仅从数据集中选取了50张图片作为验证集）

test.py文件：测试代码，可以直接运行，运行后将自动从test文件夹内读取图片进行预测，并将结果保存在result文件夹中

train.py文件：训练代码，可以直接运行，运行后代码自动开始训练，训练结束后生成模型参数文件（model_parameter.pth）和损失曲线（train result.png），保存在根目录下。另外，可以对代码进行修改，对超参数进行调整，或者将部分注释掉的代码启用以强制使用cpu或保存整个模型文件（model.pkl）

--------------------------------------

代码运行软件环境：

以下是本机运行环境，本代码应该可以运行在稍旧一些的环境下，仅供参考

Python >= 3.7.6

Cuda >= 10.2

torch >= 1.5.0

torchvision >= 0.6.0

numpy >= 1.18.1

Pillow >= 7.0.0

matplotlib >= 3.1.3
