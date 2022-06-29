from pathlib import Path
import requests
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

DATA_PATH = Path('../data')
PATH = DATA_PATH / 'mnist'

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

x_train, y_train, x_valid, y_valid = map(torch.Tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape

batch_size = 64

class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner = nn.Linear(784,10)

    def forward(self, xb):
        return self.liner(xb)


train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds,batch_size=batch_size)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds,batch_size=batch_size*2)

lr = 0.5
epochs = 2


def get_model():
    model = MnistLogistic()
    return model, optim.SGD(params=model.parameters(), lr=lr)


model, opt = get_model()
loss_func = F.cross_entropy

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        yb = yb.long()
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb.long()) for xb, yb in valid_dl)
    print(epoch,valid_loss/len(valid_dl))


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb.long())
    if opt:
       loss.backward()
       opt.step()
       opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


