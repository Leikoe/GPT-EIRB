#
#   takes a dataset generated from download.py (.txt) and creates a model
#

#!/usr/bin/env python3

import numpy as np
from model import GPTeirb
import random as rdm

def train(context_size):

    # open the whole text
    with open('./data/TinyStories.txt','r',encoding='utf-8') as f:
        text = f.read()

    model = GPTeirb([[]],[]) # create an empty model
    n = model.tokenizer.vocab_size()
    datas = np.zeros((n, n))
    Y = np.zeros(n)

    for i in range(n):
        idx = rdm.randint(0,len(text))
        Y[i] = idx
        for j in range(n):
            datas[i][j] = model(args=text[idx:idx+context_size]) # fill the data set with encoded strings
    
    data_model = GPTeirb(datas, Y)

    data_model.save('./gpteirb.bin')

train(8)

