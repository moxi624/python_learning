import torch
from torch.autograd import Variable
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt

# 假数据

n_data = torch.ones(100, 2)         # 数据的基本形态

# 分两类，一类是0，一类是1
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)

x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

# 把 x和y放到Variable的篮子里面
x,y = Variable(x), Variable(y)

# 画图
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


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


# 实例化一个神经网络，表示输入为2， 神经元为10， 输出为2
# 因为有两个类型，所以输出和输出都是2
net = Net(n_feature=2, n_hidden=10, n_output=2) # 几个类别就几个 output
print(net)

# 使用第二种方法搭建神经网络

net2 = torch.nn.Sequential(
    # 第一层 输入层
    torch.nn.Linear(2, 10),
    # 第二层 激励层
    torch.nn.ReLU(),
    #第三层 输出层
    torch.nn.Linear(10, 2)
)
print(net2);

'''
Net(
  (hidden): Linear(in_features=2, out_features=10, bias=True)   //两个输入，两个输出
  (predict): Linear(in_features=10, out_features=2, bias=True)
)
'''

# 使用优化器进行优化神经网络
# 传入参数： net.parameters():表示传入全部参数      lr=0.02：表示学习率是0.02  越低表示学习的越慢，但是越容易找到最优的值
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

#定义损失函数,是怎么计算误差的一种手段。   CrossEntropyLoss: 是计算概率的
# [0.1, 0.2, 0.7]
# [0, 0, 1]
#  CrossEntropyLoss 是计算上面两个数组的误差
loss_func = torch.nn.CrossEntropyLoss

# 开始训练,训练100步
for t in range(1000):

    # 通过神经网络训练，得到输出结果
    out = net(x)    # 可能输出的是： [-2. -0.12, 20]  这样的值    使用 F.softmax(out)转换成概率   [0.1, 0.2 , 0.7]

    # 将预测值和损失函数进行对比    prediction：预测值，  y：真实值
    loss = loss_func(out, y)

    # 将所有参数的梯度都降为0
    optimizer.zero_grad()

    # 反向传递，计算出每个节点的梯度
    loss.backward()

    # 以学习率为0.5，来优化节点的梯度
    optimizer.step()

    # 接着上面来
    if t % 2 == 0:
        plt.cla()

        # 过了一道 softmax 的激励函数后的最大概率才是预测值。  [0.1, 0.2, 0.7]  表示输出2，输出的是索引值
        prediction = torch.max(F.softmax(out), 1)[1]

        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)


# 这一步的作用，是将matplotlib变成实时打印的过程
plt.ion()
plt.show()