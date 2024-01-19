import voxelmorph2d as vm2d
import voxelmorph3d as vm3d

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import resize
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import glob
import os
from dataset import DataSet
import warnings

import cv2
import math
from PIL import Image
from torch.optim import lr_scheduler

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_dims, is_2d=False, use_gpu=False):
        self.dims = input_dims
        if is_2d:
            self.vm = vm2d
            self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, use_gpu)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu)
        self.optimizer = optim.SGD(self.voxelmorph.parameters(), lr=1e-4, momentum=0.99)
        self.params = {'batch_size': 3,
                       'shuffle': True,
                       'num_workers': 6,
                       'worker_init_fn': np.random.seed(42)
                       }
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return voxelmorph(x)

    def calculate_loss(self, y, ytrue, n=9, lamda=0.01, is_training=True):
        loss = self.vm.vox_morph_loss(y, ytrue, n, lamda)
        return loss

    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01, return_metric_score=True):
        self.optimizer.zero_grad()
        batch_fixed, batch_moving = batch_fixed.to(self.device), batch_moving.to(self.device)
        registered_image, _ = self.voxelmorph(batch_moving, batch_fixed)
        print(batch_fixed.shape)
        train_loss = self.calculate_loss(registered_image, batch_fixed, n, lamda)
        train_loss.backward()
        self.optimizer.step()
        if return_metric_score:
            train_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            return train_loss, train_dice_score
        return train_loss

    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01):
        with torch.set_grad_enabled(False):
            registered_image, _ = self.voxelmorph(batch_moving, batch_fixed)
            val_loss = self.vm.vox_morph_loss(registered_image, batch_fixed, n, lamda)
            val_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            return val_loss, val_dice_score

def savebin(arr,name):
    gray_arr = (arr - arr.min()) * 255 / (arr.max() - arr.min())
    gray_arr = gray_arr.astype('uint8')
    gray_img = Image.fromarray(gray_arr, mode='L')
#     plt.imshow(gray_arr,cmap='gray')
#     plt.show()
    gray_img.save(name)

def saveRGB(arr,name):
    cv2.imwrite(name, arr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def read_pic(fixed_path,moving_path):
    train_files_moving = glob.glob(os.path.join(moving_path, '*.dcm'))
    train_files_fixed = glob.glob(os.path.join(fixed_path, "*.dcm"))
    train_files_moving.sort()
    train_files_fixed.sort()

    DS1 = DataSet(train_files_moving)
    DS2 = DataSet(train_files_fixed)
    print("Number of training moving images: ", len(DS1))
    print("Number of training fixed images: ", len(DS2))
    moving_list = []
    fixed_list = []

    l = len(DS1)
    for i in range(l):
        moving_list.append(DS1[i])
        fixed_list.append(DS2[i])

    for i in range(l):
        c = moving_list[i].cpu().detach().numpy()
        m = c.max()
        l = c.min()
        moving_list[i] = (moving_list[i] - torch.tensor(l)) / (m - l) * 255

    for i in range(l):
        c = fixed_list[i].cpu().detach().numpy()
        m = c.max()
        l = c.min()
        fixed_list[i] = (fixed_list[i] - torch.tensor(l)) / (m - l) * 255

    moving_list1 = torch.tensor([item.cpu().detach().numpy() for item in moving_list])
    fixed_list1 = torch.tensor([item.cpu().detach().numpy() for item in fixed_list])

    custom_dataset = data.TensorDataset(moving_list1, fixed_list1)

    custom_generator = data.DataLoader(custom_dataset, batch_size=32, shuffle=True)

    return custom_dataset,custom_generator,l


fixp=input("请输入fix image路径")
movingp=input("请输入moving image路径")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
validation_set,validation_generator,length=read_pic(fixp,movingp)

vm = VoxelMorph((1, 256, 256), is_2d=True,  use_gpu=True)  # Obje
# ct of the higher level class
generator = vm.voxelmorph
model = generator.to(device)
state_dict = torch.load('300_G_new_CrossEn1.pth')
model.load_state_dict(state_dict['model'])

val_moving_test = validation_set[0:length][0].permute(0, 3, 2, 1)
val_fixed_test = validation_set[0:length][1].permute(0, 3, 2, 1)

val_moving_test = val_moving_test.cuda().float()
val_fixed_test = val_fixed_test.cuda().float()

val_registered_test, defor_test = model(val_moving_test, val_fixed_test)

for i in range(1,length+1):
    reg = val_registered_test[i - 1].cpu().data.numpy().squeeze()
    fix = val_fixed_test[i - 1].cpu().data.numpy().squeeze()

    fix_path = "./train_fix/" + str(i) + "_fixed_image_train.jpg"
    reg_path = "./train_reg/" + str(i) + "_reg_image_train.jpg"

    savebin(fix, fix_path)
    saveRGB(reg, reg_path)

