import os
import torch
import torchfile
from torch.utils.data import Dataset

from char_cnn_rnn.char_cnn_rnn import labelvec_to_onehot, labelvec_to_str


class MultimodalDataset(Dataset):
    def __init__(self, data_dir, split):
        super().__init__()
        assert split in ['train', 'val', 'test', 'trainval', 'all'], \
                'Split should be: (train|val|test|trainval|all)'

        alpha = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        self.alphabet_size = len(alpha)
        self.data_dir = data_dir

        path_manifest = os.path.join(data_dir, 'manifest.txt')
        self.files = [line.rstrip('\n') for line in open(path_manifest)]

        path_split_file = os.path.join(data_dir, split+'ids.txt')
        self.trainids = [int(line.rstrip('\n')) for line in open(path_split_file)]

        self.nclass_train = len(self.trainids)



    def __len__(self):
        # TODO: find a way to define the size of the dataset
        return 10000


    def __getitem__(self, index):
        # TODO: guarantee that each instance in batch belongs to a different class
        idx_class = torch.randint(self.nclass_train, (1,))
        fname = self.files[idx_class]

        cls_imgs = torchfile.load(os.path.join(self.data_dir, 'images', fname))
        cls_imgs = torch.Tensor(cls_imgs)
        cls_sens = torchfile.load(os.path.join(self.data_dir, 'text_c10', fname))
        cls_sens = torch.LongTensor(cls_sens)

        idx_sentence = torch.randint(cls_sens.size(2), (1,))
        idx_instance = torch.randint(cls_sens.size(0), (1,))
        idx_view = torch.randint(cls_imgs.size(2), (1,))

        img = cls_imgs[idx_instance, :, idx_view].squeeze()
        txt = cls_sens[idx_instance, :, idx_sentence].squeeze()
        txt = labelvec_to_onehot(txt)

        return img, txt
