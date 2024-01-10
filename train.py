#
#   takes a dataset generated from download.py (.txt) and creates a model
#

#!/usr/bin/env python3

import numpy as np
from model import GPTeirb
import random as rdm

def train(context_size, num_train):

    # open the whole text
    with open('./data/TinyStories.txt','r',encoding='utf-8') as f:
        text = f.read()

    model = GPTeirb([[]],[]) # create an empty model
    datas = np.zeros((num_train, num_train))
    Y = np.zeros(num_train*context_size)

    for i in range(num_train):
        idx = rdm.randint(0,len(text) - context_size)
        training_value = text[idx: idx + context_size * 5]
        # training_value_tokenised = tokenizer( training_value )[:context_size]
        # Y[i*context_size] = idx
        for j in range(1, context_size):
            Y[i * context_size + j] = (training_value_tokenised[idx:idx+j], training_value_tokenised[idx+j])
            # model(args=text[idx:idx+context_size]) # fill the data set with encoded strings

    for i in range(context_size - 1, num_train * context_size, context_size):
        value_1 = Y[i]
        for j in range(context_size - 1, num_train * context_size, context_size):
            value_2 = Y[j]
            datas[i][j] = # compare(value_1, value_2)
            

    data_model = GPTeirb(datas, Y)

    data_model.save('./gpteirb.bin')

train(8)

