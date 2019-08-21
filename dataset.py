import os
import torch
import torchfile
from torch.utils.data import Dataset

from char_cnn_rnn.char_cnn_rnn import labelvec_to_onehot



class MultimodalDataset(Dataset):
    '''
    Preprocessed Caltech-UCSD Birds 200-2011 and Oxford 102 Category Flowers
    datasets, used in ``Learning Deep Representations of Fine-grained Visual
    Descriptions``.

    Download data from: https://github.com/reedscot/cvpr2016.

    Arguments:
        data_dir (string): path to directory containing dataset files.
        split (string): which data split to load.
    '''
    def __init__(self, data_dir, split):
        super().__init__()
        possible_splits = ['train', 'val', 'test', 'trainval', 'all']
        assert split in possible_splits, \
                'Split should be: {}'.format(', '.join(possible_splits))

        path_split_classes = os.path.join(data_dir, split+'classes.txt')
        self.split_classes = \
                [line.rstrip('\n') for line in open(path_split_classes)]
        self.nclass = len(self.split_classes)

        self.data = {}
        self.num_instances = 0
        for cls in self.split_classes:
            path_imgs = os.path.join(data_dir, 'images', cls + '.t7')
            path_txts = os.path.join(data_dir, 'text_c10', cls + '.t7')
            cls_imgs = torch.Tensor(torchfile.load(path_imgs))
            cls_txts = torch.LongTensor(torchfile.load(path_txts))
            self.data[cls] = (cls_imgs, cls_txts)

            self.num_instances += cls_imgs.size(0)


    def __len__(self):
        # WARNING: this number is somewhat arbitrary, since we do not
        # necessarily use all instances in an epoch
        return self.num_instances


    def __getitem__(self, index):
        cls_id = torch.randint(self.nclass, (1,))
        cls = self.split_classes[cls_id]
        cls_imgs, cls_txts = self.data[cls]

        id_txt = torch.randint(cls_txts.size(2), (1,))
        id_instance = torch.randint(cls_txts.size(0), (1,))
        id_view = torch.randint(cls_imgs.size(2), (1,))

        img = cls_imgs[id_instance, :, id_view].squeeze()
        txt = cls_txts[id_instance, :, id_txt].squeeze()
        txt = labelvec_to_onehot(txt)

        return {'img': img, 'txt': txt}
