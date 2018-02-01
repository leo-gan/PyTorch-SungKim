import torch
import torch.nn.functional as F
from torch import  nn, optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

batch_size = 64

train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class InceptionMod(nn.Module):
    def __init__(self, in_channels):
        super(InceptionMod, self).__init__()
        self.branch_1x1_p = nn.Conv2d(in_channels,24, kernel_size=1)

        self.branch_1x1 = nn.Conv2d(in_channels,16, kernel_size=1)

        self.branch_5x5_1 = nn.Conv2d(in_channels,16, kernel_size=1)
        self.branch_5x5_2 = nn.Conv2d(16,24,kernel_size=5, padding=2)

        self.branch_3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_1x1_p = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_1x1_p = self.branch_1x1_p(branch_1x1_p)

        branch_1x1 = self.branch_1x1(x)

        branch_5x5 = self.branch_5x5_1(x)
        branch_5x5 = self.branch_5x5_2(branch_5x5)

        branch_3x3 = self.branch_3x3_1(x)
        branch_3x3 = self.branch_3x3_2(branch_3x3)
        branch_3x3 = self.branch_3x3_3(branch_3x3)

        return torch.cat([branch_1x1_p, branch_1x1, branch_5x5, branch_3x3], dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(88,20,kernel_size=5)
        self.pol  = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

        self.incep1 = InceptionMod(10)
        self.incep2 = InceptionMod(20)

    def forward(self, x):
        in_size = x.size(0)
        #print(in_size)
        x = F.relu(self.pol(self.conv1(x)))
        x = self.incep1(x)

        x = F.relu(self.pol(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size,-1)
        #print('after x = x.view(in_size,-1)', x.size(0))
        return F.log_softmax(self.fc(x))

model = Net()
criterion = nn.CrossEntropyLoss()
optimazer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, nesterov=True)
print(model)

def train(epochs):
    for epoch in range(epochs):
        print('\nEpoch:',epoch+1)
        for i, (data, label) in enumerate(train_loader):
            data, target = Variable(data), Variable(label)
            optimazer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimazer.step()

            if i%100 == 0:
                print('[{:>6}/{} ({:>02.0%})] Loss: {:.4f}]'.format(i*batch_size, len(train_loader.dataset),
                                                           i * batch_size / len(train_loader.dataset),
                                                           loss.data[0]))
def test():
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, target = Variable(data), Variable(label)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:2.0%})'.format(
        test_loss, correct, len(test_loader.dataset), correct / len(test_loader.dataset)
    ))

train(1)
test()
'''Epoch: 1
[     0/60000 (0%)] Loss: 2.3164]
[  6400/60000 (11%)] Loss: 2.3020]
[ 12800/60000 (21%)] Loss: 2.3086]
[ 19200/60000 (32%)] Loss: 2.2816]
[ 25600/60000 (43%)] Loss: 2.2922]
[ 32000/60000 (53%)] Loss: 2.2708]
[ 38400/60000 (64%)] Loss: 2.2561]
[ 44800/60000 (75%)] Loss: 2.2247]
[ 51200/60000 (85%)] Loss: 2.1518]
[ 57600/60000 (96%)] Loss: 1.7828]
Test set: Average loss: 0.0258, Accuracy: 4758/10000 (48%)
'''

