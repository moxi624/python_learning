import torch
from torch.autograd import Variable
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt

# 构造伪数据   unsqueeze() 函数的作用是增加维度， 这里是将一维数据，变成二维数据
x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)  # x data (tensor), shape=(100, 1)

#  后面加的一部分，是噪点数据
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 画图  scatter是打印散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 建立神经网络   下面是继承了 torch.nn.Module 模块
class Net(torch.nn.Module):

    # 用于该层初始化信息
    def __init__(self, n_features, n_hidden, n_output):
        # 需要初始化Net的相关信息
        super(Net, self).__init__();

        # 定义一层隐藏层信息
        # 参数：n_features: 一些信息 ，  n_hidden: 隐藏层的节点个数
        self.hidden = torch.nn.Linear(n_features, n_hidden);

        # 预测层
        self.predict = torch.nn.Linear(n_hidden, n_output);

    # 前向传递所需要的信息
    def forward(self, x):

        # 使用激活函数 relu激活，  同时x过了一层隐藏层 hidden
        x = F.relu(self.hidden(x))

        # 为什么预测函数不需要激励函数呢？
        # 因为在大多数回归问题中，预测的值分布可以在正无穷和负无穷，所以用了激励函数会使得值被截断了。
        x = self.predict(x)
        return x


# 实例化一个神经网络，表示输入为1， 神经元为10， 输出为1
net  = Net(1, 10, 1)
print(net)

# 这一步的作用，是将matplotlib变成实时打印的过程
plt.ion()
plt.show()

'''
Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
'''

# 使用优化器进行优化神经网络
# 传入参数： net.parameters():表示传入全部参数      lr=0.1：表示学习率是0.1
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

#定义损失函数,是怎么计算误差的一种手段。   MSELoss()表示均方差
loss_func = torch.nn.MSELoss()

# 开始训练,训练100步
for t in range(1000):

    # 通过神经网络训练，得到预测值
    prediction = net(x)

    # 将预测值和损失函数进行对比    prediction：预测值，  y：真实值
    loss = loss_func(prediction, y)

    # 将所有参数的梯度都降为0
    optimizer.zero_grad()

    # 反向传递，计算出每个节点的梯度
    loss.backward()

    # 以学习率为0.5，来优化节点的梯度
    optimizer.step()

    # 每五步，就进行一次打印
    if t % 5 == 0:
        plt.cla()
        # 原始的数据
        plt.scatter(x.data.numpy(), y.data.numpy())
        # 预测的数据
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # 打印出误差
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
