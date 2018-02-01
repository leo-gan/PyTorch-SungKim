import torch
from torch import nn
from torch.autograd import Variable

torch.manual_seed(123)

x = 'ihello'
y = 'helioo'

id2x = ['i', 'h', 'e', 'l', 'o']
# x_hot = {
#         'i':[1, 0, 0, 0, 0], # 'i'
#         'h':[0, 1, 0, 0, 0], # 'h'
#         'e':[0, 0, 1, 0, 0], # 'e'
#         'l':[0, 0, 0, 1, 0], # 'l'
#         'o':[0, 0, 0, 0, 1], # 'o'
# }
x2id = {c:i for i,c in enumerate(id2x)}

# input = [[x_hot[c] for c in x]]
input = [[x2id[id] for id in x]]
target = [x2id[id] for id in y]
print(len(input), input, '\n', len(target), target)

input = Variable(torch.LongTensor(input))
target = Variable(torch.LongTensor(target))

input_size = 5
num_classes = input_size
embedding_dim = 10
hidden_size = input_size
batch_size = 1
seq_len = 6
num_layers = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb = nn.Embedding(input_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        # h_0 size: batch_size,
        h_0 = Variable(torch.zeros(batch_size, num_layers, hidden_size))

        x = self.emb(x)
        x = x.view(batch_size, seq_len, -1)

        out, h_0 = self.rnn(x, h_0)
        return self.lin(out.view(-1, num_classes))


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
    res_str = [id2x[id] for id in idx]
    print('{:2} loss: {:.4} Result: {}'.format(epoch, loss.data[0], ''.join(res_str)))
    #print('{:2} loss: {:.4} Result: {}'.format(epoch, loss.data[0]), res_str)