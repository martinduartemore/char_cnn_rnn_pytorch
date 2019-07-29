import argparse
import torchfile
import torch
from collections import OrderedDict


def extract_char_cnn_rnn_weights(model_path, dataset, model_type, weights_out_path):
    '''
    Extracts weights from Char-CNN-RNN models built using Torch into
    PyTorch-compatible weights.
    '''
    assert dataset in ['birds', 'flowers'], \
            'Dataset must be (birds|flowers)'
    assert model_type in ['cvpr', 'icml'], \
            'Dataset must be (cvpr|icml)'

    model = torchfile.load(model_path)
    enc_doc = model[b'protos'][b'enc_doc']
    weights = OrderedDict()

    # convolutional segment
    # Torch storage for TemporalConvolution weights:
    #   (outputFrameSize, (kW x inputFrameSize))
    # PyTorch storage for Conv1d weights:
    #   (outputFrameSize, inputFrameSize, kW)
    conv1 = enc_doc.modules[1].modules[0]
    weights['conv1_weight'] = conv1.weight.reshape(384, 4, 70).transpose(0, 2, 1)
    weights['conv1_bias'] = conv1.bias
    conv2 = enc_doc.modules[1].modules[3]
    weights['conv2_weight'] = conv2.weight.reshape(512, 4, 384).transpose(0, 2, 1)
    weights['conv2_bias'] = conv2.bias
    conv3 = enc_doc.modules[1].modules[6]
    dim = conv3.weight.shape[0] # this changes between CVPR and ICML nets
    weights['conv3_weight'] = conv3.weight.reshape(dim, 4, 512).transpose(0, 2, 1)
    weights['conv3_bias'] = conv3.bias

    # recurrent segment
    rnn = enc_doc.modules[3]
    if model_type == 'cvpr' or dataset == 'flowers':
        rnn_i2h = rnn.modules[1].modules[0]
        weights['rnn_i2h_weight'] = rnn_i2h.weight
        weights['rnn_i2h_bias'] = rnn_i2h.bias
        rnn_h2h = rnn.modules[5].modules[0]
        weights['rnn_h2h_weight'] = rnn_h2h.weight
        weights['rnn_h2h_bias'] = rnn_h2h.bias
    elif model_type == 'icml':
        rnn_i2h = rnn.modules[1].modules[0]
        weights['rnn_i2h_weight'] = rnn_i2h.weight
        weights['rnn_i2h_bias'] = rnn_i2h.bias

        rnn_i2h_update = rnn.modules[4].modules[0]
        weights['rnn_i2h_update_weight'] = rnn_i2h_update.weight
        weights['rnn_i2h_update_bias'] = rnn_i2h_update.bias
        rnn_h2h_update = rnn.modules[5].modules[0]
        weights['rnn_h2h_update_weight'] = rnn_h2h_update.weight
        weights['rnn_h2h_update_bias'] = rnn_h2h_update.bias

        rnn_i2h_reset = rnn.modules[9].modules[0]
        weights['rnn_i2h_reset_weight'] = rnn_i2h_reset.weight
        weights['rnn_i2h_reset_bias'] = rnn_i2h_reset.bias
        rnn_h2h_reset = rnn.modules[10].modules[0]
        weights['rnn_h2h_reset_weight'] = rnn_h2h_reset.weight
        weights['rnn_h2h_reset_bias'] = rnn_h2h_reset.bias

        rnn_h2h = rnn.modules[14].modules[0]
        weights['rnn_h2h_weight'] = rnn_h2h.weight
        weights['rnn_h2h_bias'] = rnn_h2h.bias

    # embedding projection
    emb_proj = enc_doc.modules[5]
    weights['emb_proj_weight'] = emb_proj.weight
    weights['emb_proj_bias'] = emb_proj.bias

    print('Weights processed:')
    for key, value in weights.items():
        weights[key] = torch.Tensor(value)
        print('{}: {}'.format(key, value.shape))

    torch.save(weights, weights_out_path)
    print('Saved weights at: {}'.format(weights_out_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
            choices=['birds', 'flowers'],
            help='Dataset used (birds|flowers)')
    parser.add_argument('--model_type', type=str, required=True,
            choices=['icml', 'cvpr'],
            help='Model type (cvpr|icml)')
    parser.add_argument('--model_path', type=str, required=True,
            help='Path to model file in Torch format')
    parser.add_argument('--weights_out_path', type=str, required=True,
            help='Path to output weights file in PyTorch-compatible format')

    args = parser.parse_args()
    extract_char_cnn_rnn_weights(args)
