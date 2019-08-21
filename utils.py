import random
import torch
import torchfile

from char_cnn_rnn import char_cnn_rnn


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



def extract_char_cnn_rnn_weights(model_path, dataset, model_type):
    '''
    Extracts weights from Char-CNN-RNN models built using Torch into
    PyTorch-compatible weights.
    '''
    assert dataset in ['birds', 'flowers'], \
            'Dataset must be (birds|flowers)'
    assert model_type in ['cvpr', 'icml'], \
            'Dataset must be (cvpr|icml)'

    model_torch = torchfile.load(model_path)
    enc_doc = model_torch[b'protos'][b'enc_doc']
    model_pytorch = char_cnn_rnn(dataset, model_type)

    # convolutional segment
    # Torch storage for TemporalConvolution weights:
    #   (outputFrameSize, (kW x inputFrameSize))
    # PyTorch storage for Conv1d weights:
    #   (outputFrameSize, inputFrameSize, kW)
    conv1 = enc_doc.modules[1].modules[0]
    conv1_w = conv1.weight.reshape(384, 4, 70).transpose(0, 2, 1)
    model_pytorch.conv1.weight.data = torch.Tensor(conv1_w)
    model_pytorch.conv1.bias.data = torch.Tensor(conv1.bias)

    conv2 = enc_doc.modules[1].modules[3]
    conv2_w = conv2.weight.reshape(512, 4, 384).transpose(0, 2, 1)
    model_pytorch.conv2.weight.data = torch.Tensor(conv2_w)
    model_pytorch.conv2.bias.data = torch.Tensor(conv2.bias)

    conv3 = enc_doc.modules[1].modules[6]
    dim = conv3.weight.shape[0] # this changes between CVPR and ICML nets
    conv3_w = conv3.weight.reshape(dim, 4, 512).transpose(0, 2, 1)
    model_pytorch.conv3.weight.data = torch.Tensor(conv3_w)
    model_pytorch.conv3.bias.data = torch.Tensor(conv3.bias)

    # recurrent segment
    rnn = enc_doc.modules[3]
    if model_type == 'cvpr' or dataset == 'flowers':
        rnn_i2h = rnn.modules[1].modules[0]
        model_pytorch.rnn.i2h.weight.data = torch.Tensor(rnn_i2h.weight)
        model_pytorch.rnn.i2h.bias.data = torch.Tensor(rnn_i2h.bias)
        rnn_h2h = rnn.modules[5].modules[0]
        model_pytorch.rnn.h2h.weight.data = torch.Tensor(rnn_h2h.weight)
        model_pytorch.rnn.h2h.bias.data = torch.Tensor(rnn_h2h.bias)
    elif model_type == 'icml':
        rnn_i2h = rnn.modules[1].modules[0]
        model_pytorch.rnn.i2h.weight.data = torch.Tensor(rnn_i2h.weight)
        model_pytorch.rnn.i2h.bias.data = torch.Tensor(rnn_i2h.bias)

        rnn_i2h_update = rnn.modules[4].modules[0]
        model_pytorch.rnn.i2h_update.weight.data = torch.Tensor(rnn_i2h_update.weight)
        model_pytorch.rnn.i2h_update.bias.data = torch.Tensor(rnn_i2h_update.bias)
        rnn_h2h_update = rnn.modules[5].modules[0]
        model_pytorch.rnn.h2h_update.weight.data = torch.Tensor(rnn_h2h_update.weight)
        model_pytorch.rnn.h2h_update.bias.data = torch.Tensor(rnn_h2h_update.bias)

        rnn_i2h_reset = rnn.modules[9].modules[0]
        model_pytorch.rnn.i2h_reset.weight.data = torch.Tensor(rnn_i2h_reset.weight)
        model_pytorch.rnn.i2h_reset.bias.data = torch.Tensor(rnn_i2h_reset.bias)
        rnn_h2h_reset = rnn.modules[10].modules[0]
        model_pytorch.rnn.h2h_reset.weight.data = torch.Tensor(rnn_h2h_reset.weight)
        model_pytorch.rnn.h2h_reset.bias.data = torch.Tensor(rnn_h2h_reset.bias)

        rnn_h2h = rnn.modules[14].modules[0]
        model_pytorch.rnn.h2h.weight.data = torch.Tensor(rnn_h2h.weight)
        model_pytorch.rnn.h2h.bias.data = torch.Tensor(rnn_h2h.bias)

    # embedding projection
    emb_proj = enc_doc.modules[5]
    model_pytorch.emb_proj.weight.data = torch.Tensor(emb_proj.weight)
    model_pytorch.emb_proj.bias.data = torch.Tensor(emb_proj.bias)

    return model_pytorch.state_dict()
