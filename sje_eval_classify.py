import os
import random
import argparse
from collections import OrderedDict

from tqdm import tqdm

import torch
import torchfile
from torch.utils.data import DataLoader

import char_cnn_rnn as ccr



def eval_classify(net_txt, net_img, data_dir, split, num_txts_eval, batch_size, device):
    path_split_file = os.path.join(data_dir, split+'classes.txt')
    cls_list = [line.rstrip('\n') for line in open(path_split_file)]

    # prepare text data
    cls_feats_txt = []
    for cls in cls_list:
        print(cls)
        data_txt_path = os.path.join(data_dir, 'text_c10', cls + '.t7')
        data_txt = torch.LongTensor(torchfile.load(data_txt_path))

        # select T texts from all instances to represent this class
        data_txt = data_txt.permute(0, 2, 1)
        total_txts = data_txt.size(0) * data_txt.size(1)
        data_txt = data_txt.contiguous().view(total_txts, -1)
        if num_txts_eval > 0:
            num_txts_eval = min(num_txts_eval, total_txts)
            id_txts = torch.randperm(data_txt.size(0))[:num_txts_eval]
            data_txt = data_txt[id_txts]

        # TODO: adapt code to support batched version
        txt_onehot = []
        for txt in data_txt:
            txt_onehot.append(ccr.labelvec_to_onehot(txt))
        txt_onehot = torch.stack(txt_onehot)

        # separate instances into mini-batches
        # if we use a lot of text descriptions, it will not fit in gpu memory
        feats_txt = []
        for batch in torch.split(txt_onehot, batch_size, dim=0):
            with torch.no_grad():
                out = net_txt(batch.to(device))
            feats_txt.append(out)

        # average the outputs
        feats_txt = torch.cat(feats_txt, dim=0).mean(dim=0)
        cls_feats_txt.append(feats_txt)

    cls_feats_txt = torch.stack(cls_feats_txt, dim=0).to(device)

    # classification evaluation
    cls_stats = OrderedDict()
    for i, cls in enumerate(cls_list):
        # prepare image data
        data_img_path = os.path.join(data_dir, 'images', cls + '.t7')
        data_img = torch.Tensor(torchfile.load(data_img_path))
        # cub and flowers datasets have 10 image crops per instance
        # we use only the first crop per instance
        feats_img = data_img[:, :, 0].to(device)
        if net_img is not None:
            with torch.no_grad():
                feats_img = net_img(feats_img)

        scores = torch.matmul(feats_img, cls_feats_txt.t())
        max_ids = torch.argmax(scores, dim=1)
        ground_truths = torch.LongTensor(scores.size(0)).fill_(i).to(device)
        num_correct = (max_ids == ground_truths).sum().item()
        cls_stats[cls] = {'correct': num_correct, 'total': ground_truths.size(0)}

    total = sum([stats['total'] for _, stats in cls_stats.items()])
    total_correct = sum([stats['correct'] for _, stats in cls_stats.items()])
    avg_acc = total_correct / total
    return avg_acc, cls_stats



def rng_init(seed):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def main(args):
    rng_init(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_txt = ccr.char_cnn_rnn(args.dataset, args.model_type)
    net_txt.load_state_dict(torch.load(args.model_path))
    net_txt = net_txt.to(device)
    net_txt.eval()

    avg_acc, cls_stats = eval_classify(net_txt, None, args.data_dir, args.eval_split,
            args.num_txts_eval, args.batch_size, device)

    for name, stats in cls_stats.items():
        print('{:.4f}: {}'.format(stats['correct'] / stats['total'], name))
    print('Average top-1 accuracy: {:.4f}'.format(avg_acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
            choices=['birds', 'flowers'],
            help='Dataset type (birds|flowers)')
    parser.add_argument('--model_type', type=str, required=True,
            choices=['cvpr', 'icml'],
            help='Model type (cvpr|icml)')
    parser.add_argument('--data_dir', type=str, required=True,
            help='Data directory')
    parser.add_argument('--eval_split', type=str, required=True,
            choices=['train', 'val', 'test', 'trainval', 'all'],
            help='File specifying which class labels are used for training')
    parser.add_argument('--model_path', type=str, required=True,
            help='Model checkpoint path')
    parser.add_argument('--num_txts_eval', type=int,
            default=0,
            help='Number of texts to use per class (0 = use all available)')

    parser.add_argument('--batch_size', type=int,
            default=40,
            help='Evaluation batch size')
    parser.add_argument('--image_dim', type=int,
            default=1024,
            help='Image feature dimension')
    parser.add_argument('--doc_length', type=int,
            default=201,
            help='Maximum document length')
    parser.add_argument('--cnn_dim', type=int,
            default=256,
            help='Character CNN embedding dimension')

    #parser.add_argument('--gpu_id', type=int, required=True,
    #        help='Which GPU to use')
    parser.add_argument('--seed', type=int, required=True,
            help='Which RNG seed to use')

    args = parser.parse_args()
    main(args)
