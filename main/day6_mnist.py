from pathlib import Path
import requests

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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch

x_train, y_train, x_valid, y_valid = map(torch.Tensor, (x_train, y_train, x_valid, y_valid))

n, c = x_train.shape


import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)

batch_size = 64
xb = x_train[0:batch_size]
yb = y_train[0:batch_size]

out = model(xb)
loss_func = torch.nn.NLLLoss()
yb = yb.long()
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(loss_func(out, yb), accuracy(out, yb))


from IPython.core.debugger import set_trace
lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n - 1) // batch_size + 1):
        # set_trace()
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i].long()
        pred = model(xb)
        loss = loss_func(pred, yb)
        loss.backward()

        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))