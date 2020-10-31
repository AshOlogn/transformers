import torch
import torch.nn.functional as F

fname = 'data/logr_weights/bart_freq_normalized_ids.txt'
weights = []

with open(fname) as f:
    for line in f.readlines():
        w = float(line.split()[1])
        if w >= 0:
            break
        weights.append(-w)

    weights = F.softmax(torch.tensor(weights))
    print(torch.var(weights))

    weights = F.softmax(torch.tensor(weights)/10)
    print(torch.var(weights))

    weights = F.softmax(torch.tensor(weights)/1000)
    print(torch.var(weights))
    
#for i in range(len(weights)):
#    print(weights[i])


