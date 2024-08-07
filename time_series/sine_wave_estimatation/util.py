from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np


def slide(seq, look_back, hope):
    x = []
    y = []

    i = 0
    while i < len(seq) - look_back:
        x.append(seq[i:i + look_back])
        y.append([seq[i + look_back]])
        i += hope

    x = np.array(x)
    y = np.array(y)

    return x, y


def create_dataloader(x, y, batch_sz, shuffle):
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=shuffle)

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

    best_state = 0

    vld_losses = []
    trn_losses = []

    train_loss = 0.0
    valid_loss = 0.0

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

        trn_losses.append(train_loss)

        if e % valid_step != 0 and e != 0:
            vld_losses.append(valid_loss)
            continue

        mdl.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (inp, out) in enumerate(valid_dataloader):
                pred = mdl(inp)
                ls = loss(pred, out)
                valid_loss += ls.item()

        vld_losses.append(valid_loss)

        if valid_loss < min_valid_loss:
            best_state = mdl.state_dict()
            if save_path != '':
                p = save_path.format(
                    en=n_epoch,
                    vl=valid_loss,
                    tl=train_loss)
                torch.save(mdl, p)

        print('epoch: ' + str(e + 1))
        print('train loss: ' + str(train_loss / len(train_dataloader)))
        print('valid loss: ' + str(valid_loss / len(valid_dataloader)))
        print('\n')

    return trn_losses, vld_losses, best_state


