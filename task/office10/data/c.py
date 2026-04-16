import os
import cv2
import random
import numpy as np
from PIL import Image
import nibabel as nib
import SimpleITK as sitk

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, utils
import torch.utils.data as data
import json
class OfficeDataset(Dataset):
    def __init__(self, site, base_path, split='train', train_ratio=1, transform=None):
        if split == 'train':
            self.paths, self.text_labels = np.load('{}/{}_train.pkl'.format(base_path, site),
                                                   allow_pickle=True)
            total = len(self.paths)
            self.paths = self.paths[:int(0.6 * total)]
            self.text_labels = self.text_labels[:int(0.6 * total)]
        elif split == 'val':
            self.paths, self.text_labels = np.load('{}/{}_train.pkl'.format(base_path, site),
                                                   allow_pickle=True)
            total = len(self.paths)
            self.paths = self.paths[int(0.6 * total):]
            self.text_labels = self.text_labels[int(0.6 * total):]
        else:
            self.paths, self.text_labels = np.load('{}/{}_test.pkl'.format(base_path, site),
                                                   allow_pickle=True)

        label_dict = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4, 'laptop_computer': 5,
                      'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = os.path.join(base_path, site)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        self.splits = self.paths[idx].split('/')
        self.image_path = '/'.join(self.splits[2:])

        img_path = os.path.join(self.base_path, self.image_path)
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
def build_dataset(basepath, site):
    train_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    train_set = OfficeDataset(site=site, base_path=basepath, train_ratio=0.6,
                                         split='train', transform=train_transform)
    valid_set = OfficeDataset(site=site, base_path=basepath, train_ratio=0.6,
                                         split='val', transform=test_transform)
    test_set = OfficeDataset(site=site, base_path=basepath, train_ratio=0.6,
                                        split='test', transform=test_transform)
    return train_set,valid_set,test_set
if __name__ == '__main__':
    train_output = "./train/mytrain.json"
    vaild_output="./vaild/myvaild.json"
    test_output = "./test/mytest.json"
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    vaild_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    for site in ['amazon', 'caltech', 'dslr', 'webcam']:
        trainset,vaildset,testset=build_dataset("./office_caltech_10",site)
        train_cidxs=range(len(trainset))
        vaild_cidxs = range(len(vaildset))
        test_cidxs = range(len(testset))
        cname = "user{}".format(site)
        train_data['users'].append(cname)
        train_data['user_data'][cname] = {'x': [], 'y': []}
        vaild_data['users'].append(cname)
        vaild_data['user_data'][cname] = {'x': [], 'y': []}
        test_data['users'].append(cname)
        test_data['user_data'][cname] = {'x': [], 'y': []}
        #train_dids = train_cidxs
        #random.shuffle(train_dids)
        train_len = len(train_cidxs)
        train_data['num_samples'].append(train_len)
        for did in train_cidxs[:]:
            train_data['user_data'][cname]['x'].append(trainset[did][0].tolist())
            train_data['user_data'][cname]['y'].append(trainset[did][1])
        #vaild_dids = vaild_cidxs
        #random.shuffle(vaild_dids)
        vaild_len = len(vaild_cidxs)
        vaild_data['num_samples'].append(vaild_len)
        for did in vaild_cidxs[:]:
            vaild_data['user_data'][cname]['x'].append(vaildset[did][0].tolist())
            vaild_data['user_data'][cname]['y'].append(vaildset[did][1])
        #test_dids = test_cidxs
        #random.shuffle(test_dids)
        test_len = len(test_cidxs)
        test_data['num_samples'].append(test_len)
        for did in test_cidxs[:]:
            test_data['user_data'][cname]['x'].append(testset[did][0].tolist())
            test_data['user_data'][cname]['y'].append(testset[did][1])
with open(train_output,'w') as outfile:
    json.dump(train_data, outfile)
with open(vaild_output,'w') as outfile:
    json.dump(vaild_data, outfile)
with open(test_output, 'w') as outfile:
    json.dump(test_data, outfile)