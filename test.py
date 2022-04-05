"""
Written by Zachary Pulliam

Contains val() and test() fundtions to test the model.
    val() is used after training each epoch and test() is used when training has been completed.

    Parameters for testing are located in the test_args.json file.
    When ran in cmd line the args-path should be specified such as...

    python test --args-path "path/to/test_args.json"
"""

import re
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical

from rnn import RNN
from datasets import CharDataset


""" val function used after each epoch to generate text """
def val(dataset, model, seq_len):

    n = 0
    hidden_state = None
    
    index = np.random.randint(len(dataset.data)-1)
    input_seq = dataset.data[index : index+1]
    
    print('\nGenerated output from model at current state with random start...')
    print("----------------------------------------------------------------------------------------------------")

    prev = 0
    while True:
        output, hidden_state = model(input_seq, hidden_state)
        
        output = F.softmax(torch.squeeze(output[-1,:,:]), dim=0)
        dist = Categorical(output)
        index = dist.sample()

        print(dataset.intToken[index.item()], end='')
        
        input_seq[0][0] = index.item()
        n += 1
        
        if dataset.intToken[index.item()] == '\n':
            prev = 0
        else:
            prev += 1
        
        if n > seq_len or prev > 100:
            break

    print("\n----------------------------------------------------------------------------------------------------\n")


""" test function used on trained models to generate text """
def test(args):
    data_path, model_path, hidden_size, num_layers, seq_len, device, primer = \
        args['data_path'], args['model_path'], args['hidden_size'], args['num_layers'], args['seq_len'], args['device'], args['primer']

    dataset = CharDataset(data_path, device)

    model = RNN(dataset.vocab_size, dataset.vocab_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_path))
    hidden_state = None
    
    primer_embed = []
    for i, l in enumerate(primer):
        primer_embed.append(torch.tensor([[dataset.tokenInt[l]]]))

    for i in primer_embed:
        output, hidden_state = model(i, hidden_state)


    output = F.softmax(torch.squeeze(output[-1,:,:]), dim=0)
    dist = Categorical(output)
    index = dist.sample()

    print("\n----------------------------------------------------------------------------------------------------\n")
    print('Generated output from model using primer {0}...'.format(primer))
    print(primer, end='')

    print(dataset.intToken[index.item()], end='')
    
    input_seq = torch.tensor([[index.item()]])
    
    n, prev = 0, 0
    while True:
        output, hidden_state = model(input_seq, hidden_state)
        
        output = F.softmax(torch.squeeze(output), dim=0)
        dist = Categorical(output)
        index = dist.sample()

        print(dataset.intToken[index.item()], end='')

        input_seq[0][0] = index.item()
        n += 1
        
        if dataset.intToken[index.item()] == '\n':
            prev = 0
        else:
            prev += 1

        if n > seq_len or prev > 100:
            break

    print("\n----------------------------------------------------------------------------------------------------\n")
   

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, default='', help='test_args.json path')

    opt = parser.parse_args()
    with open(opt.args_path, 'r') as f:
        args = json.load(f)
    return args


if __name__ == "__main__":
    args = parse_opt()
    test(args)