"""
Written by Zachary Pulliam

Contains Character dataset class to create a dataset from a text file.
Contains create_loader() function to create a simple dataloader from the dataset.
"""

import torch

from random import shuffle


""" Creates a simple dataloader from one of the two datasets below """
def create_loader(data, start, seq_len):
    loader = []
    while True:
        if start + seq_len + 1 > len(data):
            break
        else:
            input_seq = data[start : start+seq_len]
            target_seq = data[start+1 : start+seq_len+1]
            loader.append([input_seq, target_seq])
            start += seq_len
    shuffle(loader)
    return loader
        

""" Character based dataset """
class CharDataset():
    def __init__(self, path, device):
        self.data = open(path, 'r').read()
        chars = sorted(set(''.join(self.data)))
        self.data_size, self.vocab_size = len(self.data), len(chars)

        self.intToken = dict(enumerate(chars))
        self.tokenInt = {token: index for index, token in self.intToken.items()} 

        self.data = list(self.data)
        for i, ch in enumerate(self.data):
            self.data[i] = self.tokenInt[ch]

        self.data = torch.tensor(self.data).to(device)
        self.data = torch.unsqueeze(self.data, dim=1)