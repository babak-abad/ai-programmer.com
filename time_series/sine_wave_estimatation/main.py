import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
import util as utl

win_sz = 15
step_x = 0.1
n_batch = 8
n_epoch = 1000
hid_sz = 10
lr = 0.1
hope = 2

trn_sz = 0.6
vld_sz = 0.1

data = [step_x*x for x in range(0, int(10*np.pi/step_x))]

data = (np.sin(data) + 1)/2.0
x, y = utl.slide(data, win_sz, hope)

trn_sz = int(len(data) * trn_sz)
vld_sz = int(len(data) * vld_sz)

trn_data = data[0: trn_sz]
vld_data = data[trn_sz: trn_sz+vld_sz]
tst_data = data[trn_sz+vld_sz:]

plt.plot(y)

trn_dl = utl.create_dataloader(
    *utl.slide(trn_data, win_sz, hope),
    n_batch,
    True)
vld_dl = utl.create_dataloader(
    *utl.slide(data, win_sz, hope),
    n_batch,
    True)
tst_dl = utl.create_dataloader(
    *utl.slide(data, win_sz, hope),
    n_batch,
    False)

model = nn.Sequential(
    nn.Linear(win_sz, win_sz),
    nn.Sigmoid(),
    nn.Linear(win_sz, 1),
    nn.Sigmoid()
)

loss = nn.MSELoss()
opt = Adam(model.parameters(), lr=lr)
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
plt.plot(y.numpy())
plt.show()
