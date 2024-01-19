import os
import pydicom
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms as T
import SimpleITK as sitk
import pandas as pd


class DataSet(Dataset):
     def __init__(self,files):
          self.files=files

     def __len__(self):
          return len(self.files)

     def __getitem__(self, index):
          # img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
          moving_data = Image.fromarray(pydicom.dcmread(self.files[index]).pixel_array)
          # fixed_data = Image.fromarray(pydicom.dcmread(self.fixed_files[index]).pixel_array)
          transform=  transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((256,256))])
          moving_data = transform(moving_data)
          # fixed_data = transform(fixed_data)
          return moving_data
