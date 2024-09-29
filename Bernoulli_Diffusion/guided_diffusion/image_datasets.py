import math
import random
from pathlib import Path
from PIL import Image
import blobfile as bf
import numpy as np
import torch as th
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
import os
import torch
import sys

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    print('datadir', data_dir)

    classes = None

    dataset = DataLoader(ImageDataset("/content/2", 128, True), batch_size=batch_size,shuffle=False, num_workers=4, pin_memory=True)

    print('lenloader', len(loader))
   # return loader
    while True:
        yield from loader

class ImageDataset(Dataset):
    def __init__(self, folder_path, max_length, random_crop=True):
        self.folder_path = folder_path
        self.max_length = max_length
        self.random_crop = random_crop
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.preprocessed_data = []
        for file in self.file_list:
          file_path = os.path.join(folder_path, file)
          data = np.load(file_path, mmap_mode='r')
          self.preprocessed_data.append(self._preprocess(data))

    def __len__(self):
        return len(self.file_list)

    def _preprocess(self, data):
      if data.shape[0] > self.max_length:
        if self.random_crop:
          start = np.random.randint(0, data.shape[0] - self.max_length)
          data = data[start:start+self.max_length, :]
        else:
          data = data[:self.max_length, :]
      elif data.shape[0] < self.max_length:
        pad_width = ((0, self.max_length - data.shape[0]), (0, 0))
        data = np.pad(data, pad_width, mode='constant')
      data = torch.from_numpy(data).float()
      data = data.unsqueeze(0)
      return data

    def __getitem__(self, idx):
      return self.preprocessed_data[idx]
