import torch
import torch.nn as nn

from .net_modules.fixed_rnn import fixed_rnn
from .net_modules.fixed_gru import fixed_gru



class char_cnn_rnn(nn.Module):
    '''
    Char-CNN-RNN model, described in ``Learning Deep Representations of
    Fine-grained Visual Descriptions``.

    This implementation supports two distinct incarnations of the models for
    the birds and flowers datasets:
        * ``Learning Deep Representations of Fine-Grained Visual Descriptions
            (https://github.com/reedscot/cvpr2016)
        * ``Generative Adversarial Text-to-Image Synthesis``
            (https://github.com/reedscot/icml2016)

    Each incarnation contains slight variations on the model architecture, and
    the architecture also varies depending on the dataset used.

    Arguments:
        dataset (string): which dataset was used.
        model_type (string): which incarnation of the model to use.
    '''
    def __init__(self, dataset, model_type):
        super().__init__()
        assert dataset in ['birds', 'flowers'], \
                'dataset should be (birds|flowers)'
        assert model_type in ['cvpr', 'icml'], \
                'model_type should be (cvpr|icml)'

        if model_type == 'cvpr':
            rnn_dim = 256
            use_maxpool3 = True
            rnn = fixed_rnn
            rnn_num_steps = 8
        else:
            rnn_dim = 512
            if dataset == 'flowers':
                use_maxpool3 = True
                rnn = fixed_rnn
                rnn_num_steps = 8
            else:
                use_maxpool3 = False
                rnn = fixed_gru
                rnn_num_steps = 18

        self.dataset = dataset
        self.model_type = model_type
        self.use_maxpool3 = use_maxpool3

        # network setup
        # (B, 70, 201)
        self.conv1 = nn.Conv1d(70, 384, kernel_size=4)
        self.threshold1 = nn.Threshold(1e-6, 0)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        # (B, 384, 66)
        self.conv2 = nn.Conv1d(384, 512, kernel_size=4)
        self.threshold2 = nn.Threshold(1e-6, 0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        # (B, 512, 21)
        self.conv3 = nn.Conv1d(512, rnn_dim, kernel_size=4)
        self.threshold3 = nn.Threshold(1e-6, 0)
        if use_maxpool3:
            self.maxpool3 = nn.MaxPool1d(kernel_size=3,stride=2)
        # (B, rnn_dim, rnn_num_steps)
        self.rnn = rnn(num_steps=rnn_num_steps, emb_dim=rnn_dim)
        # (B, rnn_dim)
        self.emb_proj = nn.Linear(rnn_dim, 1024)
        # (B, 1024)


    def forward(self, txt):
        # temporal convolutions
        out = self.conv1(txt)
        out = self.threshold1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.threshold2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.threshold3(out)
        if self.use_maxpool3:
            out = self.maxpool3(out)

        # recurrent computation
        out = out.permute(0, 2, 1)
        out = self.rnn(out)

        # linear projection
        out = self.emb_proj(out)

        return out



def prepare_text(string, max_str_len=201):
    '''
    Converts a text description from string format to one-hot tensor format.
    '''
    labels = str_to_labelvec(string, max_str_len)
    one_hot = labelvec_to_onehot(labels)
    return one_hot



def str_to_labelvec(string, max_str_len):
    string = string.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    alpha_to_num = {k:v+1 for k,v in zip(alphabet, range(len(alphabet)))}
    labels = torch.zeros(max_str_len).long()
    max_i = min(max_str_len, len(string))
    for i in range(max_i):
        labels[i] = alpha_to_num.get(string[i], alpha_to_num[' '])

    return labels



def labelvec_to_onehot(labels):
    labels = torch.LongTensor(labels).unsqueeze(1)
    one_hot = torch.zeros(labels.size(0), 71).scatter_(1, labels, 1.)
    # ignore zeros in one-hot mask (position 0 = empty one-hot)
    one_hot = one_hot[:, 1:]
    one_hot = one_hot.permute(1,0)
    return one_hot



def onehot_to_labelvec(tensor):
    labels = torch.zeros(tensor.size(1), dtype=torch.long)
    val, idx = torch.nonzero(tensor).split(1, dim=1)
    labels[idx] = val+1
    return labels



def labelvec_to_str(labels):
    '''
    Converts a text description from one-hot tensor format to string format.
    '''
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    string = [alphabet[x-1] for x in labels if x > 0]
    string = ''.join(string)
    return string
