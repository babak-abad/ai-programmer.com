import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.optim import Adam
import util as utl
import configs as cfg

data = [cfg.step_x*x for x in range(0, int(10*np.pi/cfg.step_x))]

data = (np.sin(data) + 1)/2.0
x, y = utl.slide(data, cfg.win_sz, cfg.hope)

trn_sz = int(len(data) * cfg.trn_sz)
vld_sz = int(len(data) * cfg.vld_sz)

trn_data = data[0: trn_sz]
vld_data = data[trn_sz: trn_sz+vld_sz]
tst_data = data[trn_sz+vld_sz:]

trn_dl = utl.create_dataloader(
    *utl.slide(trn_data, cfg.win_sz, cfg.hope),
    cfg.n_batch,
    True)
vld_dl = utl.create_dataloader(
    *utl.slide(data, cfg.win_sz, cfg.hope),
    cfg.n_batch,
    True)
tst_dl = utl.create_dataloader(
    *utl.slide(data, cfg.win_sz, cfg.hope),
    cfg.n_batch,
    False)

model = nn.Sequential(
    nn.Linear(cfg.win_sz, cfg.hidden_sz),
    nn.Sigmoid(),
    nn.Linear(cfg.hidden_sz, 1),
    nn.Sigmoid()
)

loss = nn.MSELoss()
opt = Adam(model.parameters(), lr=cfg.lr)
model.train()

trn_ls, vld_ls, bst_state = utl.train(
    mdl=model,
    train_dataloader=trn_dl,
    valid_dataloader=vld_dl,
    n_epoch=cfg.n_epoch,
    opt=opt,
    loss=loss,
    valid_step=cfg.vld_step,
    save_path='')

y_act = []
y_pred = []

model.eval()
with torch.inference_mode():
    for i, (inp, out) in enumerate(tst_dl):
        y = out.numpy().reshape((1, -1))[0].tolist()
        y_act.extend(y)

        y = model(inp)
        y = y.numpy().reshape(1, -1)[0].tolist()
        y_pred.extend(y)

plt.plot(y_act)
plt.plot(y_pred)
plt.legend(['actual', 'predicted'])
plt.title('actual vs predicted values')
plt.show()

plt.plot(trn_ls)
plt.plot(vld_ls)
plt.legend(['train loss', 'validation loss'])
plt.title('training procedure')
plt.show()
