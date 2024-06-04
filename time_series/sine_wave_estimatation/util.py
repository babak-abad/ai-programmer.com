from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import numpy as np
from torch import nn
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt


def create_dataloader(x, y, batch_sz, shuffle):
    tensor_x = torch.Tensor(x)  # transform to torch tensor
    tensor_y = torch.Tensor(y.reshape((-1, 1)))

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=shuffle)  # create your dataloader
    return dataloader


def train(
        mdl,
        train_dataloader,
        valid_dataloader,
        n_epoch,
        opt,
        loss,
        valid_step,
        save_path):
    min_valid_loss = float('inf')
    mdl = mdl.cpu()

    best_state = 0

    for e in range(0, n_epoch):
        mdl.train()
        train_loss = 0.0
        for i, (inp, out) in enumerate(train_dataloader):
            opt.zero_grad()
            pred = mdl(inp)
            ls = loss(pred, out)
            ls.backward()
            opt.step()
            train_loss += ls.item()

        if e % valid_step != 0 and e != 0:
            continue

        mdl.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (inp, out) in enumerate(valid_dataloader):
                pred = mdl(inp)
                ls = loss(pred, out)
                valid_loss += ls.item()

        if valid_loss < min_valid_loss and save_path != '':
            p = save_path.format(
                en=n_epoch,
                vl=valid_loss,
                tl=train_loss)
            torch.save(mdl, p)

            best_state = mdl.state_dict()

        print('epoch: ' + str(e + 1))
        print('train loss: ' + str(train_loss / len(train_dataloader)))
        print('valid loss: ' + str(valid_loss / len(valid_dataloader)))
        print('\n')

    return best_state


# if idx = 0 then it prints x
# if idx = 1 then it prints y
def draw_dataloader(dt, idx):
    res = []
    for v in dt.dataset:
        print(v[0])
        print(v[1])
        res.extend(v[idx].tolist())
    res = np.array(res)
    plt.plot(res)
    plt.show()


def normalize(data, rng=(0, 1)):
    mms = MinMaxScaler(feature_range=rng)
    scaler = mms.fit(data)
    return scaler.transform(data)

#
# def create_dataloader(data, win_sz):
#     x = []
#     y = []
#
#     for i in range(0, len(data) - win_sz - 1):
#         xx = data[i:win_sz + i, 1].astype('float32')
#         yy = data[win_sz + i, 1].astype('float32')
#         x.append(xx)
#         y.append(yy)
#
#     x = np.array(x)
#     y = np.array(y).reshape(-1, 1)
#     return x, y


def slide(seq, win_sz, hope):
    x = []
    y = []

    i = 0
    while i < len(seq) - win_sz:
        x.append(seq[i:i+win_sz])
        y.append(seq[i+win_sz])
        i += hope

    x = np.array(x)
    y = np.array(y)

    return x, y


class Data_Provider:
    def __init__(self, data, win_sz):
        self.x = []
        self.y = []

        dt = data.reshape(-1, 1)

        self.mms = MinMaxScaler(feature_range=(0, 1))
        s = self.mms.fit(dt)
        norm_data = s.transform(dt)
        norm_data = norm_data.reshape(1, -1)[0]
        for i in range(0, len(norm_data) - win_sz - 1):
            xx = norm_data[i:win_sz + i].astype('float32')
            yy = norm_data[win_sz + i].astype('float32')
            self.x.append(xx)
            self.y.append(yy)

        self.x = np.array(self.x)
        self.y = np.array(self.y).reshape(-1, 1)

    def denormalize(self, x):
        t = x.reshape(-1, 1)
        t = self.mms.inverse_transform(t)
        t = t.reshape(1, -1)[0]
        return t

    def get_dataloader(self, batch_sz):
        return create_dataloader(
            self.x,
            self.y,
            batch_sz,
            True)



    # mms = MinMaxScaler(feature_range=(0, 1))
    # scaler =  mms.fit(trn)
    # normalized_trn = scaler.transform(trn)
    #
    # scaler = mms.fit(vld)
    # normalized_vld = scaler.transform(vld)
    #
    # scaler = mms.fit(tst)
    # normalized_tst = scaler.transform(tst)
