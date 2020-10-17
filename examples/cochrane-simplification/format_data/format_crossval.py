import json
import random
import os
from transformers import BertTokenizer
import sys

def get_abstract(article):
    return ' '.join([x['text'] for x in article['abstract']])

def get_pls(article):
    return article['pls'] if article['pls_type'] == 'long' else ' '.join([x['text'] for x in article['pls']])

max_length=int(sys.argv[1])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def length_filter(article):
    return len(tokenizer.encode(get_abstract(article))) <= max_length and len(tokenizer.encode(get_pls(article))) <= max_length

def tokenize(text):
    return tokenizer.batch_encode_plus(text,
                                       max_length=max_length,
                                       pad_to_max_length=True)['input_ids']


data = json.load(open('data/data_final.json'))
data = list(filter(length_filter, data))

abstracts = [get_abstract(a) for a in data]
pls = [get_pls(a) for a in data]

X = tokenize(abstracts)
X.extend(tokenize(pls))

y = ['0']*len(data)
y.extend(['1']*len(data))

#os.mkdir('data/cls-crossval-truncated-512-inf')
with open('data/cls-crossval-truncated-512-inf/source.txt', 'w') as f:
    for tokens in X:
        f.write(' '.join([str(t) for t in tokens]) + '\n')

with open('data/cls-crossval-truncated-512-inf/target.txt', 'w') as f:
    f.write('\n'.join(y))
