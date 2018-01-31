import torch
import torch.nn.functional as F
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

class LinModel(torch.nn.Module):
    def __init__(self):
        super(LinModel, self).__init__()
        self.lin = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = F.sigmoid(self.lin(x))
        return y_pred

model = LinModel()
criterion = torch.nn.BCELoss()
optim = torch.optim.SGD(params=model.parameters(), lr= 0.008)

for epoch in range(500):
    optim.zero_grad()
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    loss.backward()
    optim.step()
    print(epoch, loss.data[0])

print(5, model(Variable(torch.Tensor([[5]]))))