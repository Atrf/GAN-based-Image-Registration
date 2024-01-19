import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1) # 输入大小 [batch_size, 2, 256, 256]，输出大小 [batch_size, 16, 256, 256]
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 输入大小 [batch_size, 16, 128, 128]，输出大小 [batch_size, 32, 256, 256]
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # 输入大小 [batch_size, 32, 64, 64]，输出大小 [batch_size, 64, 256, 256]
        self.conv4 = nn.Conv2d(64 ,64 ,3 ,padding = 1)
        self.pool = nn.MaxPool2d(2, 2) # 输入大小 [batch_size, 64, 32, 32]，输出大小 [batch_size, 64, 16, 16]
        self.fc1 = nn.Linear(16384, 256) # 输入大小 [batch_size, 64*64*64]，输出大小 [batch_size, 256]
        self.fc2 = nn.Linear(256, 2) # 输入大小 [batch_size, 256]，输出大小 [batch_size, 2]
        self.norm1 = nn.LayerNorm(16384)
        self.norm2 = nn.LayerNorm(256)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x))) # 卷积后，使用ReLU函数激活，池化后输出到下一层，大小为[batch_size, 16, 128, 128]
        x = self.pool(nn.functional.relu(self.conv2(x))) # 卷积后，使用ReLU函数激活，池化后输出到下一层，大小为[batch_size, 32, 64, 64]
        x = self.pool(nn.functional.relu(self.conv3(x))) # 卷积后，使用ReLU函数激活，池化后输出到下一层，大小为[batch_size, 64, 32, 32]
        x = self.pool(nn.functional.relu(self.conv4(x)))
        
        x = x.view(-1, 16384) # 将特征图展开成向量，大小为[batch_size, 64*64*64]
        x = self.norm1(x)
        x = nn.functional.relu(self.fc1(x)) # 全连接层，使用ReLU函数激活，输出到下一层，大小为[batch_size, 256]
        x = self.norm2(x)
        x = self.fc2(x) # 全连接层，输出大小为[batch_size, 2]
        return x