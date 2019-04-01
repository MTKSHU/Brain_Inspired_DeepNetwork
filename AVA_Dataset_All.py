#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:23:01 2019

@author: debo
"""
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd

# %%
path = "/home/debo/dataset/AVA/"


class AVA_Dataset_All(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_files = [x for x in os.listdir(self.root_dir) if '.jpg' in x]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_files[idx]))

        if self.transform:
            img = self.transform(img)

        return img


class AVA_Dataset_Complementary_Colors(Dataset):
    def __init__(self, root_dir, train=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = []
        self.files = []
        if train == True:
            labels = open(self.root_dir + "AVA_dataset/style_image_lists/train.lab").readlines()
            labels = [int(l.strip('\n')) for l in labels]
            self.labels = [0 if x != 1 else 1 for x in labels]
            files = open(self.root_dir + "AVA_dataset/style_image_lists/train.jpgl").readlines()
            self.files = [f.strip('\n') for f in files]
        else:
            labels = open(self.root_dir + "AVA_dataset/style_image_lists/test.multilab").readlines()
            labels = [int(l.strip('\n').split(' ')[0]) for l in labels]
            self.labels = [0 if x != 1 else 1 for x in labels]
            files = open(self.root_dir + "AVA_dataset/style_image_lists/test.jpgl").readlines()
            self.files = [f.strip('\n') for f in files]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, "resized_images_320_320_padded_white/" + self.files[idx]))
        if self.transform:
            img = self.transform(img)
# path = "/home/debopriyo/dataset/AVA/resized_images_320_320_padded_white/"
# dataset = AVA_Dataset_All(root_dir=path)

# print(dataset[0])
