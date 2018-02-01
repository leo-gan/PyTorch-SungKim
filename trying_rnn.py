import torch
from torch import nn
from torch.autograd import Variable

torch.manual_seed(123)

id2x = ['i', 'h', 'e', 'l', 'o']
x_hot = {
        'i':[1, 0, 0, 0, 0], # 'i'
        'h':[0, 1, 0, 0, 0], # 'h'
        'e':[0, 0, 1, 0, 0], # 'e'
        'l':[0, 0, 0, 1, 0], # 'l'
        'o':[0, 0, 0, 0, 1], # 'o'
}

x = 'ihello'
y = 'helioo'

input = [[x_hot[c] for c in x]]

target = [1,2,3,0,4,4]
print(len(input), input, '\n', len(target), target)

input = Variable(torch.Tensor(input))
target = Variable(torch.LongTensor(target))

input_size = 5
num_classes = input_size
hidden_size = input_size
batch_size = 1
seq_len = 6
num_layers = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, x):
        # h_0 size: batch_size,
        h_0 = Variable(torch.zeros(batch_size, num_layers, hidden_size))
        x.view(batch_size, seq_len, input_size)

        out, h_0 = self.rnn(x, h_0)
        return out.view(-1, num_classes)

model = Net()
print(model)

criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)

for epoch in range(100):
    opt.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    opt.step()

    _, out_id = output.max(1)
    idx = out_id.data.numpy()
    res_str = [id2x[id] for id in idx.squeeze()]
    print('{:2} loss: {:.4} Result: {}'.format(epoch, loss.data[0], ''.join(res_str)))
    #print('{:2} loss: {:.4} Result: {}'.format(epoch, loss.data[0]), res_str)