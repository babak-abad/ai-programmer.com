from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import numpy as np


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

    return trn_losses, vld_losses, best_state


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