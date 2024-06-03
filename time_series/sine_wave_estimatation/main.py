import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.optim import Adam

x = np.linspace(0, 1, 50).reshape((-1, 1)).astype('float32')
y = np.power(x, 2).reshape((-1, 1)).astype('float32')

x = torch.tensor(x)
y = torch.tensor(y)

model = nn.Sequential(
    nn.LSTM(1, 5, fi),
    nn.Sigmoid(),
    nn.Linear(1, 1),
    nn.Sigmoid()
)

loss = nn.MSELoss()
opt = Adam(model.parameters(), lr=0.1)
model.train()
n_batch = 6
n_epoch = 1000

for i in range(n_epoch):
    for b in range(0, len(x), n_batch):
        inp = x[b:b+n_batch]
        out = y[b:b+n_batch]
        opt.zero_grad()
        pred = model(inp)
        ls = loss(pred, out)
        ls.backward()
        opt.step()

model.eval()
with torch.inference_mode():
    x = torch.tensor(np.linspace(0, 1, 50).astype('float32').reshape((-1, 1)))
    y = model(x)

plt.plot(x, y)
plt.show()





