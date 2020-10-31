import torch
from torch.nn.functional import softmax
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import sys
import random
import os
from os import path


def get_abstract(article):
    return ' '.join([x['text'] for x in article['abstract']])

def get_pls(article):
    return article['pls'] if article['pls_type'] == 'long' else ' '.join([x['text'] for x in article['pls']])

def mask_batch(tokens, tokenizer, num_mask):
    indexed_tokens = []
    mask_indices = []
    
    for i in range(10):
        cur_mask_indices = random.sample(list(range(1,len(tokens)-1)), num_mask)
        masked_tokens = [tokens[index] for index in cur_mask_indices]
        
        for index in cur_mask_indices:
            tokens[index] = '[MASK]'
        
        indexed_tokens.append(tokenizer.convert_tokens_to_ids(tokens))
        mask_indices.append(cur_mask_indices)
        
        for j in range(num_mask):
            index = cur_mask_indices[j]
            tokens[index] = masked_tokens[j]

    return indexed_tokens, mask_indices


def run_model_sentence(tokens, tokenizer, model, num_mask=5, batch_size=1, device='cuda'):
    (indexed_tokens, mask_indices) = mask_batch(tokens, tokenizer, num_mask)
    predictions = []

    model.eval()
    model.to(device)
    
    start_index = 0
    while start_index < len(indexed_tokens):
        end_index = min(start_index + batch_size, len(indexed_tokens))
        cur_indexed_tokens = torch.tensor(indexed_tokens[start_index:end_index], dtype=torch.long).to(device)
        segment_ids = torch.ones((end_index-start_index, len(tokens)), dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model.forward(cur_indexed_tokens, token_type_ids=segment_ids)
            predictions.append(outputs[0].to('cpu'))

        start_index = end_index
    
    predictions = torch.cat(predictions, dim=0)
    return predictions, mask_indices


def eval_sentence(sentence, tokenizer, model, num_mask=5, batch_size=1, device='cuda'):
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > 510:
        tokens = tokens[:510]
    
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    #if num_mask is a float, treat as a percentage of tokens to mask, 
    #of course excluding the CLS and SEP tokens
    if type(num_mask) == float:
        num_mask = int(num_mask * (len(tokens)-2))
    
    (predictions, mask_indices) = run_model_sentence(tokens, tokenizer, model, num_mask, batch_size, device)
    probabilities = []
    
    for i in range(len(predictions)):
        for mask_index in mask_indices[i]:
            distribution = softmax(predictions[i, mask_index], dim=0)
            masked_token_index = tokenizer.convert_tokens_to_ids(tokens[mask_index])
            prob = distribution[masked_token_index].item()
            probabilities.append(prob)
    
    return probabilities

def eval_paragraph(paragraph, tokenizer, model, num_mask=5, batch_size=1, device='cuda'):
    probabilities = []
    for sent in sent_tokenize(paragraph):
        if type(num_mask) == int and len(tokenizer.tokenize(sent)) < num_mask:
            print('skipping sentence...')
            continue
        probabilities += eval_sentence(sent, tokenizer, model, num_mask, batch_size, device)
     
    return probabilities


def eval_article(article, tokenizer, model, num_mask=5, batch_size=1, device='cuda'):
    abstract_probs = eval_paragraph(article['abstract'], tokenizer, model, num_mask, batch_size, device) 
    pls_probs = eval_paragraph(article['pls'], tokenizer, model, num_mask, batch_size, device)
    gen_probs = eval_paragraph(article['gen'], tokenizer, model, num_mask, batch_size, device)
    
    return abstract_probs, pls_probs, gen_probs


def probability_results(data, input_file_name, tokenizer, model, file_name, num_mask=5, batch_size=1, device='cuda'):

    prefix = path.split(input_file_name)[-2]

    #read in the dois already processed (if the file_name exists) so that they 
    #can be ignored in this run
    already = set()
    if path.isfile(path.join(prefix, file_name)):
        with open(path.join(prefix, file_name)) as f:
            for l in f.readlines():
                if len(l) > 0:
                    already.add(l.split(', ')[0])

    for index,article in enumerate(data):
        if article['doi'] in already:
            continue
        
        print(index)
        
        (abstract_probs, pls_probs, gen_probs) = eval_article(article, tokenizer, model, num_mask, batch_size, device)
        abstract_avg = np.mean(abstract_probs)
        pls_avg = np.mean(pls_probs)
        gen_avg = np.mean(gen_probs)
        
        with open(path.join(prefix, file_name), 'a+', 1) as f:
            f.write(f'{article["doi"]} {abstract_avg} {pls_avg} {gen_avg}\n')
            f.flush()


model_name = sys.argv[1]
input_file_name = sys.argv[2]
file_name = sys.argv[3]
num_mask = float(sys.argv[4]) if '.' in sys.argv[4] else int(sys.argv[4])

batch_size = int(sys.argv[5])
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

start_index = int(sys.argv[6])
end_index = int(sys.argv[7])

print(input_file_name)

sys.exit()

data = json.load(open(input_file_name))
probability_results(data[start_index:end_index], input_file_name, tokenizer, model, file_name, num_mask, batch_size, device='cuda')

