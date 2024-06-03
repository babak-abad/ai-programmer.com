import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import util as utl

x = np.linspace(0, 1, 50).reshape((-1, 1)).astype('float32')
y = np.power(x, 2).reshape((-1, 1)).astype('float32')

n_batch = 6
n_epoch = 1000

trn_dl = utl.create_dataloader(x, y, n_batch, True)
vld_dl = utl.create_dataloader(x, y, n_batch, True)
tst_dl = utl.create_dataloader(x, y, n_batch, True)

model = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid(),
    nn.Linear(1, 1),
    nn.Sigmoid()
)

loss = nn.MSELoss()
opt = Adam(model.parameters(), lr=0.1)
model.train()

utl.train(
    mdl=model,
    train_dataloader=trn_dl,
    valid_dataloader=vld_dl,
    n_epoch=n_epoch,
    opt=opt,
    loss=loss,
    valid_step=100,
    save_path='')

model.eval()
with torch.inference_mode():
    x = torch.tensor(np.linspace(0, 1, 50).astype('float32').reshape((-1, 1)))
    y = model(x)

plt.plot(x, y)
plt.show()
