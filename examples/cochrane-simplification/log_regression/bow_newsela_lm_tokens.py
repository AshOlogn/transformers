import json
import os
from os.path import join
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertConfig, BartTokenizer


def make_vector(text, tokenizer):
    token_ids = tokenizer.encode(text)[1:-1]
    count_vector = np.zeros(tokenizer.vocab_size, dtype=np.int16)
    for ID in token_ids:
        count_vector[ID] += 1
    return count_vector


def dataloader(data_dir, batch_size=5000):

    names = [x[:-6] for x in os.listdir(data_dir) if x[-5:] == '3.txt']
    index = 0

    while index < len(names):
        cur_names = names[index:index+batch_size]
        tuples = []
        
        for name in cur_names:
            hard = open(join(data_dir, f'{name}.0.txt')).read()
            simple = open(join(data_dir, f'{name}.3.txt')).read()
            tuples.append((hard, simple))

        yield tuples
        index += batch_size


def construct_dataset(tuples, tokenizer):

    X = np.empty((2*len(tuples), tokenizer.vocab_size), dtype=np.int16)
    y = np.empty(2*len(tuples), dtype=np.int16)

    index = 0
    for s,t in tuples:
        X[index] = make_vector(s, tokenizer)
        X[index+1] = make_vector(t, tokenizer)
        y[index] = 0
        y[index+1] = 1
        index += 2

    return X, y

def get_vocab(tokenizer):
    tokens = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(tokenizer.vocab_size)]
    return tokens

def simple_term_counts(data_dir='data/newsela/articles'):

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
    model = LogisticRegression(max_iter=100)

    for batch in dataloader(data_dir):
        X, y = construct_dataset(batch, tokenizer)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

        #apply feature scaling
        #X_train = normalize(X_train)
        #X_test = normalize(X_test)

        #model.fit(X_train, y_train)
        #predictions = model.predict(X_test)
        #print(accuracy_score(y_test, predictions))

        X = normalize(X)
        model.fit(X, y)

    vocab = get_vocab(tokenizer)
    weights = np.squeeze(model.coef_, axis=0).tolist()

    sorted_weights = filter(lambda x: len(x[1].strip()) > 0, zip(range(tokenizer.vocab_size), vocab, weights))
    sorted_weights = list(sorted(sorted_weights, key=lambda x: x[2]))
    
    with open('data/logr_weights/bart_freq_newsela_ids.txt', 'w') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{ID} {weight}\n')

    with open('data/logr_weights/bart_freq_newsela_tokens.txt', 'w') as f:
        for ID, word, weight in sorted_weights:
            f.write(f'{word} {weight}\n')

print(simple_term_counts())
