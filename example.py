import argparse
import torch

from dataset import MultimodalDataset

import char_cnn_rnn as ccr
from utils import extract_char_cnn_rnn_weights



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # extract weights from original models
    net_state_dict = extract_char_cnn_rnn_weights(args.torch_model_path,
            args.dataset, args.model_type)
    # save weights for later and load with torch.load()
    #torch.save(net_state_dict, args.weights_out_path)

    # create Char-CNN-RNN model and load weights
    net = ccr.char_cnn_rnn(args.dataset, args.model_type)
    #net.load_state_dict(torch.load(args.weights_out_path))
    net.load_state_dict(net_state_dict)
    net = net.to(device)
    net.eval()
    print(net)

    # prepare text and run it through model
    # default maximum text length is 201 characters (truncated after that)
    txt = 'Text description here'
    txt = ccr.prepare_text(txt)
    txt = txt.unsqueeze(0).to(device)
    out = net(txt)
    print(out.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
            choices=['birds', 'flowers'],
            help='Dataset type')
    parser.add_argument('--model_type', type=str, required=True,
            choices=['cvpr', 'icml'],
            help='Model type')
    parser.add_argument('--torch_model_path', type=str, required=True,
            help='Path to original Torch model')
    #parser.add_argument('--weights_out_path', type=str, required=True,
    #        help='Path to save extracted weights')

    args = parser.parse_args()
    main(args)
