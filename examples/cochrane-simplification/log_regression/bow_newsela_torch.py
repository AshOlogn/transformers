import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BartTokenizer
import os
from os.path import join
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score

class LogR(nn.Module):
    def __init__(self, dim):
        super(LogR, self).__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

    def predict(self, x):
        output = (self(x) >= 0.5).int()
        return output


def make_vector(text, tokenizer):
    token_ids = tokenizer.encode(text, verbose=False)[1:-1]
    count_vector = np.zeros(tokenizer.vocab_size)
    for ID in token_ids:
        count_vector[ID] += 1
    return count_vector

def sanitize_article(text):
    return '\n'.join([line for line in text.split('\n') if len(line) > 0 and line[0]!='#'])

def dataloader(data_dir, indices=None, batch_size=32):

    names = [x[:-6] for x in os.listdir(data_dir) if x[-5:] == '3.txt']
    indices = list(range(len(names))) if indices is None else list(indices)
    random.shuffle(indices)

    index = 0
    while index < len(indices):
        cur_names = [names[i] for i in indices[index:index+batch_size]]
        tuples = []

        for name in cur_names:
            hard = sanitize_article(open(join(data_dir, f'{name}.0.txt')).read())
            simple = sanitize_article(open(join(data_dir, f'{name}.3.txt')).read())
            tuples.append((hard, simple))

        yield tuples
        index += batch_size

def construct_dataset(tuples, tokenizer):

    X = np.empty((2*len(tuples), tokenizer.vocab_size), dtype=np.float)
    y = np.arange(2*len(tuples), dtype=np.float) % 2

    index = 0
    for s,t in tuples:
        X[index] = make_vector(s, tokenizer)
        X[index+1] = make_vector(t, tokenizer)
        index += 2

    return X, y

def get_vocab(tokenizer):
    tokens = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(tokenizer.vocab_size)]
    return tokens


def train_newsela(data_dir):

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
    loss_func = nn.BCELoss(reduction='mean')
    model = LogR(tokenizer.vocab_size).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(5):
        print('beginning epoch ' + str(i+1))

        #first train on train split
        for batch in dataloader(data_dir, batch_size=16):

            X, y = construct_dataset(batch, tokenizer)
            X = torch.from_numpy(X).float().to('cuda')
            y = torch.from_numpy(y).float().to('cuda')

            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()

    return model

def get_weights(model, tokenizer, weights_dir):
    vocab = get_vocab(tokenizer)
    weights = model.linear.weight.squeeze(0).cpu().tolist()
    sorted_weights = filter(lambda x: len(x[1].strip()) > 0, zip(range(tokenizer.vocab_size), vocab, weights))
    sorted_weights = list(sorted(sorted_weights, key=lambda x: x[2]))

    with open(join(weights_dir, 'newsela_ids.txt'), 'w') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{ID} {weight}\n')

    with open(join(weights_dir, 'newsela_tokens.txt'), 'w') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{word} {weight}\n')


model = train_newsela('data/newsela/articles')
torch.save(model.state_dict(), 'models/')




