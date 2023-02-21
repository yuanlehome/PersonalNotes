import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor
# 一、导入分布式专用 Fleet API
from paddle.distributed import fleet
# 构建分布式数据加载器所需 API
from paddle.io import DataLoader, DistributedBatchSampler
# 设置 GPU 环境
paddle.set_device('gpu')

class MyNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(MyNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3))

        self.flatten = paddle.nn.Flatten()

        self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
        self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

epoch_num = 10
batch_size = 32
learning_rate = 0.001
val_acc_history = []
val_loss_history = []

def train():
    # 二、初始化 Fleet 环境
    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

    model = MyNet(num_classes=10)
    # 三、构建分布式训练使用的网络模型
    
    model = fleet.distributed_model(model)

    opt = paddle.optimizer.Adam(learning_rate=learning_rate,parameters=model.parameters())
    # 四、构建分布式训练使用的优化器
    opt = fleet.distributed_optimizer(opt)

    transform = ToTensor()
    cifar10_train = paddle.vision.datasets.Cifar10(mode='train',
                                           transform=transform)
    cifar10_test = paddle.vision.datasets.Cifar10(mode='test',
                                          transform=transform)

    # 五、构建分布式训练使用的数据集
    train_sampler = DistributedBatchSampler(cifar10_train, batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(cifar10_train, batch_sampler=train_sampler, num_workers=2)

    valid_sampler = DistributedBatchSampler(cifar10_test, batch_size, drop_last=True)
    valid_loader = DataLoader(cifar10_test, batch_sampler=valid_sampler, num_workers=2)


    for epoch in range(epoch_num):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)

            if batch_id % 1000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc = paddle.metric.accuracy(logits, y_data)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)

if __name__ == "__main__":
    train()