"""
Written by Zachary Pulliam

Contains train() function used to train the RNN() model

    Parameters for training are located in the train_args.json file.
    When ran in cmd line the args-path should be specified such as...

    python train --args-path "path/to/train_args.json"
"""

import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch import nn, optim

from rnn import RNN
from test import val
from datasets import CharDataset, create_loader


""" Function used to train RNN... parameters in train_args.txt """
def train(args):
    data_path, save_path, hidden_size, num_layers, seq_len, lr, epochs, val_size, device = \
        args['data_path'], args['save_path'], args['hidden_size'], args['num_layers'], args['seq_len'], args['lr'], args['epochs'], args['val_size'], args['device']

    dataset = CharDataset(data_path, device)

    model = RNN(dataset.vocab_size, dataset.vocab_size, hidden_size, num_layers).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('\nBeginning training...\n')

    for epoch in range(1, epochs+1):
        print('Epoch: {0}'.format(epoch), end='\t')
        
        start = np.random.randint(seq_len)
        loader = create_loader(dataset.data, start, seq_len)
        print('Text loaded in from random starting point...')

        running_loss = []
        hidden_state = None
        
        for _, [x, y] in enumerate(tqdm(loader)):

            output, hidden_state = model(x, hidden_state)

            loss = loss_func(torch.squeeze(output), torch.squeeze(y))
            running_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print("Epoch: {0} of {1} \t Average Loss: {2:.8f} \t Final Loss: {3:.8f}".format(epoch, epochs, sum(running_loss)/len(running_loss), running_loss[-1]))
        torch.save(model.state_dict(), save_path)
        
        val(dataset, model, val_size)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str, default='', help='train_args.json path')

    opt = parser.parse_args()
    with open(opt.args_path, 'r') as f:
        args = json.load(f)
    with open(args['save_path'][0:-3] + 'json', 'w') as g:
        json.dump(args, g, indent=2)
    return args


if __name__ == "__main__":
    args = parse_opt()
    train(args)