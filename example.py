import argparse
import torch
import char_cnn_rnn as ccr

from dataset import MultimodalDataset



def main(args):
    # extract weights from original models using
    # ccr.extract_char_cnn_rnn_weights()
    # check function for details

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create an instance of the Char-CNN-RNN model and load extracted weights from file
    net = ccr.char_cnn_rnn(args.dataset, args.model_type)
    net.load_weights_from_file(args.weights_path)
    net = net.to(device)
    #print(net)

    # prepare text and run it through model
    txt = 'Text description here'
    txt = ccr.prepare_text(txt, max_str_len=201)
    txt = txt.unsqueeze(0).to(device)
    out = net(txt)
    print(out.shape)

    loader = MultimodalDataset('/A/martin/datasets/birds_dataset/cvpr2016_cub',
            'trainval')
    img, txt = loader[0]
    test1 = ccr.onehot_to_labelvec(txt)
    test2 = ccr.labelvec_to_str(test1)
    print(test2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
            choices=['birds', 'flowers'],
            help='Dataset type (birds|flowers)')
    parser.add_argument('--model_type', type=str, required=True,
            choices=['cvpr', 'icml'],
            help='Model type (cvpr|icml)')
    parser.add_argument('--weights_path', type=str, required=True,
            help='Path to model weights file')

    args = parser.parse_args()
    main(args)
