import torch.utils.data as data
import scipy.io
from PIL import Image
import numpy as np

import os

class AircraftDataset(data.Dataset):

    def __init__(self, root_dir="/daiyaanarfeen/data/fgvc-aircraft-2013b/data", train=True, transforms=None):
        self.root_dir = root_dir
        if train:
            self.examples = open(os.path.join(root_dir, 'images_train.txt'), 'r').readlines()
            self.examples += open(os.path.join(root_dir, 'images_val.txt'), 'r').readlines()
            self.labels = dict([l.replace('\n', '').split(' ', 1) for l in open(os.path.join(root_dir, 'images_variant_trainval.txt'), 'r')]) 
        else:
            self.examples = open(os.path.join(root_dir, 'images_test.txt'), 'r').readlines()
            self.labels = dict([l.replace('\n', '').split(' ', 1) for l in open(os.path.join(root_dir, 'images_variant_test.txt'), 'r')]) 
        self.examples = [e.replace('\n', '') for e in self.examples]
        self.label_to_id = open(os.path.join(root_dir, 'variants.txt'), 'r').readlines()
        self.label_to_id = dict([(l.replace('\n', ''), i) for (i, l) in enumerate(self.label_to_id)])
        self.transform = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.examples[idx] + '.jpg')
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[self.examples[idx]]
        label = self.label_to_id[label]

        return img, label
