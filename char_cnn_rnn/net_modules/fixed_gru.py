import torch
import torch.nn as nn



class fixed_gru(nn.Module):
    '''
    Custom GRU that matches the implementation provided by Reed et al.
    '''
    def __init__(self, num_steps, emb_dim):
        super().__init__()
        # update gate
        self.i2h_update = nn.Linear(emb_dim, emb_dim)
        self.h2h_update = nn.Linear(emb_dim, emb_dim)

        # reset gate
        self.i2h_reset = nn.Linear(emb_dim, emb_dim)
        self.h2h_reset = nn.Linear(emb_dim, emb_dim)

        # candidate hidden state
        self.i2h = nn.Linear(emb_dim, emb_dim)
        self.h2h = nn.Linear(emb_dim, emb_dim)

        self.num_steps = num_steps


    def forward(self, txt):
        res = []
        res_intermediate = []
        for i in range(self.num_steps):
            if i == 0:
                output = torch.tanh(self.i2h(txt[:, i])).unsqueeze(1)
            else:
                # compute update and reset gates
                update = torch.sigmoid(self.i2h_update(txt[:, i]) + \
                        self.h2h_update(res[i-1]))
                reset = torch.sigmoid(self.i2h_reset(txt[:, i]) + \
                        self.h2h_reset(res[i-1]))

                # compute candidate hidden state
                gated_hidden = reset * res[i-1]
                p1 = self.i2h(txt[:, i])
                p2 = self.h2h(gated_hidden)
                hidden_cand = torch.tanh(p1 + p2)

                # use gates to interpolate hidden state
                zh = update * hidden_cand
                zhm1 = ((update * -1) + 1) * res[i-1]
                output = zh + zhm1

            res.append(output)
            res_intermediate.append(output)

        res = torch.cat(res, dim=1)
        res = torch.mean(res, dim=1)
        return res
