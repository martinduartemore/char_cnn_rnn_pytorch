import random
import torch



def rng_init(seed):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias, a=-0.08, b=0.08)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.uniform_(m.weight, a=-0.08, b=0.08)
        torch.nn.init.uniform_(m.bias, a=-0.08, b=0.08)
