import torch
import torch.nn as nn



class fixed_rnn(nn.Module):
    '''
    Custom RNN that matches the implementation provided by Reed et al.

    We cannot use PyTorch's RNN because Reed's implementation does not use
    bias for the hidden layer (b_{hh}) during the computation of the first
    time step.
    '''
    def __init__(self, num_steps, emb_dim):
        super().__init__()
        self.i2h = nn.Linear(emb_dim, emb_dim)
        self.h2h = nn.Linear(emb_dim, emb_dim)

        self.num_steps = num_steps
        self.relu = torch.nn.functional.relu


    def forward(self, txt):
        res = []
        for i in range(self.num_steps):
            i2h = self.i2h(txt[:, i]).unsqueeze(1)
            if i == 0:
                output = self.relu(i2h)
            else:
                h2h = self.h2h(res[i-1])
                output = self.relu(i2h + h2h)
            res.append(output)

        res = torch.cat(res, dim=1)
        res = torch.mean(res, dim=1)
        return res
