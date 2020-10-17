import json
import random
import os
from transformers import BartTokenizer
import sys

def get_abstract(article):
    return ' '.join([x['text'] for x in article['abstract']])

def get_pls(article):
    return article['pls'] if article['pls_type'] == 'long' else ' '.join([x['text'] for x in article['pls']])

def get_frac(a, p, tokenizer, max_length):
    return len(tokenizer.encode(p, max_length=max_length))/len(tokenizer.encode(a, max_length=max_length))

threshold=float(sys.argv[1])
max_length=int(sys.argv[2])
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def length_filter(article):
    return len(tokenizer.encode(get_abstract(article))) <= max_length and len(tokenizer.encode(get_pls(article))) <= max_length

data = list(filter(length_filter, json.load(open('data/data_final.json'))))
random.shuffle(data)

num_train = int(0.8 * len(data))
num_val = 1
num_test = len(data)-num_train-num_val

base_dir = f'data/truncated-{max_length}-{threshold}'
#base_dir = 'data/dummy'


if not os.path.isdir(base_dir):
    os.system(f'mkdir -p {base_dir}')
elif len(os.listdir(base_dir)) > 0:
    os.system(f'rm {base_dir}/*')

with open(f'{base_dir}/train.source', 'w') as f:
    with open(f'{base_dir}/train.target', 'w') as g:
        with open(f'{base_dir}/train.doi', 'w') as h:
            for article in data[:num_train]:
                abstract = get_abstract(article)
                pls = get_pls(article)
                if get_frac(abstract, pls, tokenizer, max_length) <= threshold:
                    f.write(get_abstract(article).replace('\n', ' ')+'\n')
                    g.write(get_pls(article).replace('\n', ' ')+'\n')
                    h.write(article['doi']+'\n')

with open(f'{base_dir}/val.source', 'w') as f:
    with open(f'{base_dir}/val.target', 'w') as g:
        with open(f'{base_dir}/val.doi', 'w') as h:
            for article in data[num_train:num_train+num_val]:
                abstract = get_abstract(article)
                pls = get_pls(article)
                if get_frac(abstract, pls, tokenizer, max_length) <= threshold:
                    f.write(get_abstract(article).replace('\n', ' ')+'\n')
                    g.write(get_pls(article).replace('\n', ' ')+'\n')
                    h.write(article['doi']+'\n')

with open(f'{base_dir}/test.source', 'w') as f:
    with open(f'{base_dir}/test.target', 'w') as g:
        with open(f'{base_dir}/test.doi', 'w') as h:
            for article in data[num_train+num_val:]:
                abstract = get_abstract(article)
                pls = get_pls(article)
                if get_frac(abstract, pls, tokenizer, max_length) <= threshold:
                    f.write(get_abstract(article).replace('\n', ' ')+'\n')
                    g.write(get_pls(article).replace('\n', ' ')+'\n')
                    h.write(article['doi']+'\n')
