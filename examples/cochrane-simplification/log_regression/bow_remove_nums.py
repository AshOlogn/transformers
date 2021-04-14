import json
from os.path import join
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertConfig

def get_abstract(article):
    return ' '.join([x['text'] for x in article['abstract']])

def get_pls(article):
    return article['pls'] if article['pls_type'] == 'long' else ' '.join([x['text'] for x in article['pls']])

def construct_dataset(data_dir):

    data = json.load(open(data_dir))
    shuffle(data)
    
    X = []
    y = []

    for article in data:
        X.extend([get_abstract(article), get_pls(article)])
        y.extend([0,1])
    
    return X, y

def get_vocab(X, remove_numbers=False):
    vectorizer = CountVectorizer()
    vectorizer.fit(X)
    vocab = vectorizer.vocabulary_
    
    def no_digits(word):
        return (not any(ch.isdigit() for ch in word) or word.lower().islower())

    return [word for word in vocab if (no_digits(word) or not remove_numbers)]

def list_index(l, indices):
    return [l[i] for i in indices]

def simple_term_counts(data_dir='data/data_final.json', remove_numbers=False, k=5):

    X, y = construct_dataset(data_dir)
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = []

    for i,(train_indices, test_indices) in enumerate(splitter.split(X, y)):

        print(f'beginning fold {i}...')
        
        train_indices = train_indices.tolist()
        test_indices = test_indices.tolist()

        X_train = list_index(X, train_indices)
        y_train = list_index(y, train_indices)
        X_test = list_index(X, test_indices)
        y_test = list_index(y, test_indices)

        vectorizer = CountVectorizer()

        # get vocabulary that will be used
        vocab = get_vocab(X_train, remove_numbers)
        vectorizer = CountVectorizer(vocabulary=vocab)

        #vectorize the data
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_test = vectorizer.fit_transform(X_test).toarray()

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

    return np.mean(accuracies) 
        

def simple_tfidf(data_dir='data/data_final.json', remove_numbers=False, k=5):

    X, y = construct_dataset(data_dir)
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = []

    for i,(train_indices, test_indices) in enumerate(splitter.split(X, y)):

        print(f'beginning fold {i}')

        train_indices = train_indices.tolist()
        test_indices = test_indices.tolist()

        X_train = list_index(X, train_indices)
        y_train = list_index(y, train_indices)
        X_test = list_index(X, test_indices)
        y_test = list_index(y, test_indices)

        vectorizer = TfidfVectorizer()

        # get vocabulary that will be used
        vocab = get_vocab(X_train, remove_numbers)
        vectorizer = TfidfVectorizer(vocabulary=vocab)

        #vectorize the data
        X_train = vectorizer.fit_transform(X_train).toarray()
        X_test = vectorizer.fit_transform(X_test).toarray()

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

    return np.mean(accuracies)


print(simple_tfidf(remove_numbers=True))



