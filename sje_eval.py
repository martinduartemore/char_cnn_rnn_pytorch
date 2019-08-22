import os
import random
import argparse
from collections import OrderedDict

from tqdm import tqdm

import torch
import torchfile
from torch.utils.data import DataLoader

from utils import rng_init
import char_cnn_rnn as ccr



def encode_data(net_txt, net_img, data_dir, split, num_txts_eval, batch_size, device):
    '''
    Encoder for preprocessed Caltech-UCSD Birds 200-2011 and Oxford 102
    Category Flowers datasets, used in ``Learning Deep Representations of
    Fine-grained Visual Descriptions``.

    Warning: if you decide to not use all sentences (i.e., num_txts_eval > 0),
    sentences will be randomly sampled and their features will be averaged to
    provide a class representation. This means that the evaluation procedures
    should be performed multiple times (using different seeds) to account for
    this randomness.

    Arguments:
        net_txt (torch.nn.Module): text processing network.
        net_img (torch.nn.Module): image processing network.
        data_dir (string): path to directory containing dataset files.
        split (string): which data split to load.
        num_txts_eval (int): number of textual descriptions to use for each
            class. The embeddings are averaged per-class.
        batch_size (int): batch size to split data processing into chunks.
        device (torch.device): which device to do computation in.

    Returns:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.
    '''
    path_split_file = os.path.join(data_dir, split+'classes.txt')
    cls_list = [line.rstrip('\n') for line in open(path_split_file)]

    cls_feats_img = []
    cls_feats_txt = []
    for cls in cls_list:
        # prepare image data
        data_img_path = os.path.join(data_dir, 'images', cls + '.t7')
        data_img = torch.Tensor(torchfile.load(data_img_path))
        # cub and flowers datasets have 10 image crops per instance
        # we use only the first crop per instance
        feats_img = data_img[:, :, 0].to(device)
        if net_img is not None:
            with torch.no_grad():
                feats_img = net_img(feats_img)
        cls_feats_img.append(feats_img)

        # prepare text data
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

        # convert to one-hot tensor to run through network
        # TODO: adapt code to support batched version
        txt_onehot = []
        for txt in data_txt:
            txt_onehot.append(ccr.labelvec_to_onehot(txt))
        txt_onehot = torch.stack(txt_onehot)

        # if we use a lot of text descriptions, it will not fit in gpu memory
        # separate instances into mini-batches to process them using gpu
        feats_txt = []
        for batch in torch.split(txt_onehot, batch_size, dim=0):
            with torch.no_grad():
                out = net_txt(batch.to(device))
            feats_txt.append(out)

        # average the outputs
        feats_txt = torch.cat(feats_txt, dim=0).mean(dim=0)
        cls_feats_txt.append(feats_txt)

    cls_feats_txt = torch.stack(cls_feats_txt, dim=0)

    return cls_feats_img, cls_feats_txt, cls_list



def eval_classify(cls_feats_img, cls_feats_txt, cls_list):
    '''
    Classification evaluation.

    Arguments:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.

    Returns:
        avg_acc (float): percentage of correct classifications for all classes.
        cls_stats (OrderedDict): dictionary whose keys are class names and each
            entry is a dictionary containing the 'total' of images for the
            class and the number of 'correct' classifications.
    '''
    cls_stats = OrderedDict()
    for i, cls in enumerate(cls_list):
        feats_img = cls_feats_img[i]
        scores = torch.matmul(feats_img, cls_feats_txt.t())
        max_ids = torch.argmax(scores, dim=1).to('cpu')
        ground_truths = torch.LongTensor(scores.size(0)).fill_(i)
        num_correct = (max_ids == ground_truths).sum().item()
        cls_stats[cls] = {'correct': num_correct, 'total': ground_truths.size(0)}

    total = sum([stats['total'] for _, stats in cls_stats.items()])
    total_correct = sum([stats['correct'] for _, stats in cls_stats.items()])
    avg_acc = total_correct / total
    return avg_acc, cls_stats



def eval_retrieval(cls_feats_img, cls_feats_txt, cls_list, k_values=[1,5,10,50]):
    '''
    Retrieval evaluation (Average Precision).

    Arguments:
        cls_feats_img (list of torch.Tensor): list containing precomputed image
            features for each image, separated by class.
        cls_feats_txt (torch.Tensor): tensor containing precomputed (and
            averaged) textual features for each class.
        cls_list (list of string): list of class names.
        k_values (list, optional): list of k-values to use for evaluation.

    Returns:
        map_at_k (OrderedDict): dictionary whose keys are the k_values and the
            values are the mean Average Precision (mAP) for all classes.
        cls_stats (OrderedDict): dictionary whose keys are class names and each
            entry is a dictionary whose keys are the k_values and the values
            are the Average Precision (AP) per class.
    '''
    total_num_cls = cls_feats_txt.size(0)
    total_num_img = sum([feats.size(0) for feats in cls_feats_img])
    scores = torch.zeros(total_num_cls, total_num_img)
    matches = torch.zeros(total_num_cls, total_num_img)

    for i, cls in enumerate(cls_list):
        start_id = 0
        for j, feats_img in enumerate(cls_feats_img):
            end_id = start_id + feats_img.size(0)
            scores[i, start_id:end_id] = torch.matmul(feats_img, cls_feats_txt[i])
            if i == j: matches[i, start_id:end_id] = 1
            start_id = start_id + feats_img.size(0)

    for i, s in enumerate(scores):
        _, inds = torch.sort(s, descending=True)
        matches[i] = matches[i, inds]

    map_at_k = OrderedDict()
    for k in k_values:
        map_at_k[k] = torch.mean(matches[:, 0:k]).item()

    cls_stats = OrderedDict()
    for i, cls in enumerate(cls_list):
        ap_at_k = OrderedDict()
        for k in k_values:
            ap_at_k[k] = torch.mean(matches[i, 0:k]).item()
        cls_stats[cls] = ap_at_k

    return map_at_k, cls_stats



def main(args):
    rng_init(args.seed)
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    net_txt = ccr.char_cnn_rnn(args.dataset, args.model_type)
    net_txt.load_state_dict(torch.load(args.model_path, map_location=device))
    net_txt = net_txt.to(device)
    net_txt.eval()

    cls_feats_img, cls_feats_txt, cls_list = encode_data(net_txt, None, args.data_dir,
            args.eval_split, args.num_txts_eval, args.batch_size, device)

    mean_ap, cls_stats = eval_retrieval(cls_feats_img, cls_feats_txt, cls_list)
    print('----- RETRIEVAL -----')
    if args.print_class_stats:
        print('  PER CLASS:')
        for name, stats in cls_stats.items():
            print(name)
            for k, ap in stats.items():
                print('{:.4f}: AP@{}'.format(ap, k))
        print()

    print('  mAP:')
    for k, v in mean_ap.items():
        print('{:.4f}: mAP@{}'.format(v, k))
    print('---------------------')
    print()

    avg_acc, cls_stats = eval_classify(cls_feats_img, cls_feats_txt, cls_list)
    print('--- CLASSIFICATION --')
    if args.print_class_stats:
        print('  PER CLASS:')
        for name, stats in cls_stats.items():
            print('{:.4f}: {}'.format(stats['correct'] / stats['total'], name))
        print()

    print('Average top-1 accuracy: {:.4f}'.format(avg_acc))
    print('---------------------')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
            choices=['birds', 'flowers'],
            help='Dataset type')
    parser.add_argument('--model_type', type=str, required=True,
            choices=['cvpr', 'icml'],
            help='Model type')
    parser.add_argument('--data_dir', type=str, required=True,
            help='Data directory')
    parser.add_argument('--eval_split', type=str, required=True,
            choices=['train', 'val', 'test', 'trainval', 'all'],
            help='Which dataset split to use')

    parser.add_argument('--model_path', type=str, required=True,
            help='Model checkpoint path')
    parser.add_argument('--num_txts_eval', type=int,
            default=0,
            help='Number of texts to use per class (0 = use all)')
    parser.add_argument('--print_class_stats', type=bool,
            default=True,
            help='Whether to print per class statistics or not')
    parser.add_argument('--batch_size', type=int,
            default=40,
            help='Evaluation batch size')

    parser.add_argument('--seed', type=int, required=True,
            help='Which RNG seed to use')
    parser.add_argument('--use_gpu', type=bool,
            default=True,
            help='Whether or not to use GPU')

    args = parser.parse_args()
    main(args)
