from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import random
import unicodedata
import string
import torch
import torch.nn as nn
import time
import math


def find_files(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicode2ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)


cate_names = {}
all_cate = []


def read_names(file_name):
    names = open(file_name, encoding='utf8').read().strip().split('\n')
    return [unicode2ascii(name) for name in names]


for file_name in find_files('../data/names/*.txt'):
    category = os.path.splitext(os.path.basename(file_name))[0]
    all_cate.append(category)
    names = read_names(file_name)
    cate_names[category] = names

n_cate = len(all_cate)


def letter2index(letter):
    return all_letters.index(letter)


def letter2tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter2index(letter)] = 1
    return tensor


def names2tensor(names):
    tensor = torch.zeros(1, len(names), n_letters)
    for ni, letter in enumerate(names):
        tensor[0][ni][letter2index(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        r_out,(h_n,h_c) = self.lstm(x, None)
        output = self.out(r_out[:, -1, :])
        output = self.softmax(output)
        return output



n_hidden = 128
rnn = RNN(n_letters, n_hidden, 2, n_cate)

input = names2tensor('zhang')
output = rnn(input)
print(output)


def category_from_output(output):
    top_n,top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_cate[category_i], category_i


print(category_from_output(output))

exit(0)

def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_train_example():
    category = random_choice(all_cate)
    name = random_choice(cate_names[category])
    category_tensor = torch.tensor([all_cate.index(category)], dtype=torch.long)
    name_tensor = names2tensor(name)
    return category, name, category_tensor, name_tensor

#
# for i in range(10):
#     category, name, category_tensor, name_tensor = random_train_example()
#     print(category,name)


learning_rate = 0.005
criterion = nn.NLLLoss()


def train(category_tensor, name_tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    for i in range(name_tensor.size()[0]):
        output, hidden = rnn.forward(name_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()
all_losses = []
current_loss = 0
n_iters = 100000
print_every = 5000
plot_every = 1000


for iter in range(n_iters):
    category, name, category_tensor, name_tensor = random_train_example()
    output, loss = train(category_tensor, name_tensor)
    current_loss += loss
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, name, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
    current_loss = 0

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.figure()
plt.plot(all_losses)
plt.show()

confusion = torch.zeros(n_cate, n_cate)
n_confusion = 10000


def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_train_example()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_cate.index(category)
    confusion[category_i][guess_i] += 1


for i in range(n_cate):
    confusion[i] = confusion[i] / confusion[i].sum()


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_cate, rotation=90)
ax.set_yticklabels([''] + all_cate)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()

