import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
import util as utl

step_x = 0.1
data = [step_x*x for x in range(0, int(8*np.pi/step_x))]

data = (np.sin(data) + 1)/2.0
win_sz = 15
x, y = utl.slide(data, win_sz, 2)

plt.plot(y)
n_batch = 8
n_epoch = 1000

trn_dl = utl.create_dataloader(x, y, n_batch, True)
vld_dl = utl.create_dataloader(x, y, n_batch, True)
tst_dl = utl.create_dataloader(x, y, n_batch, True)

hid_sz = 10

model = nn.Sequential(
    nn.Linear(win_sz, win_sz),
    nn.Sigmoid(),
    nn.Linear(win_sz, 1),
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
    y = model(torch.tensor(x.astype('float32')))

plt.legend(['actual', 'predicted'])
plt.legend('sss')
plt.plot(y.numpy())
plt.show()
