import json
from os.path import join


def make_json(data_dir, fname, outname, prefix='train'):
    
    gens = [x.strip() for x in open(fname) if (len(x) > 5 and x[:5] != '-----')]
    dois = [x.strip() for x in open(join(data_dir, f'{prefix}.doi')).readlines() if len(x.strip()) > 0][:len(gens)]
    abstracts = [x.strip() for x in open(join(data_dir, f'{prefix}.source')).readlines() if len(x.strip()) > 0][:len(gens)]
    pls = [x.strip() for x in open(join(data_dir, f'{prefix}.target')).readlines() if len(x.strip()) > 0][:len(gens)]

    data = []
    for d,a,p,g in zip(dois, abstracts, pls, gens):
        article = {'doi': d, 'abstract': a, 'pls': p, 'gen': g}
        data.append(article)

    open(outname, 'w').write(json.dumps(data, indent=2))


data_dir = 'data/truncated-512-inf'

model_dir = 'bart-ul_facebook-bart-large-xsum_lr-def_epochs-5_maxlen-512-truncated_bs-1_ul-sp_ul-num-weights-100_ul-alpha-10_output'
make_json(data_dir, join(model_dir,'gen_train_1_0-1780_text_only.txt'), join(model_dir, 'gen_train.json'))


model_dir = 'bart-ul_facebook-bart-large-xsum_lr-def_epochs-5_maxlen-512-truncated_bs-1_ul-sp_ul-num-weights-100_ul-alpha-1000_output'
make_json(data_dir, join(model_dir,'gen_train_1_0-1780_text_only.txt'), join(model_dir, 'gen_train.json'))


model_dir = 'bart-ul_facebook-bart-large-xsum_lr-def_epochs-5_maxlen-512-truncated_bs-1_ul-sp_ul-num-weights-100_ul-alpha-1000000_output'
make_json(data_dir, join(model_dir,'gen_train_1_0-1780_text_only.txt'), join(model_dir, 'gen_train.json'))

