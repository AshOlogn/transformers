import os
from os.path import join
import json


def foo(dirname, fnames, outname):
    data = []
    for fname in fnames:
        data.extend(json.load(open(join(dirname, fname))))

    with open(join(dirname, outname), 'w') as f:
        f.write(json.dumps(data, indent=2))


dirname = 'ull_both_5_100'

fnames = [
    "gen_nucleus_test_1_0-125.json",
    "gen_nucleus_test_1_125-315.json",
    "gen_nucleus_test_1_315-500.json",
]

outname = "gen_nucleus_test_1_0-500.json"

foo(dirname, fnames, outname)




