import torch.utils.data as data
import scipy.io
from PIL import Image
import numpy as np

import os

class FlowersDataset(data.Dataset):

    def __init__(self, root_dir="/daiyaanarfeen/data/flowers", train=True, transforms=None):
        self.root_dir = root_dir
        splits = scipy.io.loadmat(os.path.join(root_dir, 'setid.mat'))
        if train:
            self.examples = np.hstack((splits['trnid'][0], splits['valid'][0])) 
        else:
            self.examples = splits['tstid'][0]
        self.samples = self.examples
        self.labels = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))['labels'][0]
        self.transform = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'jpg', 'image_' + str(self.examples[idx]).zfill(5) + '.jpg')
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[self.examples[idx] - 1] - 1

        return img, label
