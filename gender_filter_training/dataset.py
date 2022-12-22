import torchvision.transforms as tf
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random 
import glob
import torch
import os

class mydata_set(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.male_list = glob.glob(f'{self.data_path}/M/*.*')
        self.female_list = glob.glob(f'{self.data_path}/F/*.*')
       
        self.male_len = len(self.male_list)
        self.female_len = len(self.female_list)
        self.transform = tf.Compose([tf.Resize((256,256)), tf.ToTensor(), tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    def __len__(self):
        return min([self.male_len, self.female_len])

    def __getitem__(self, idx):

        p = random.random()
       
        if p >=0.5:
            img= Image.open(self.male_list[idx]).convert('RGB')
            tensor = self.transform(img)
            label = torch.tensor([1.], requires_grad=True)
        
         
        elif p < 0.5:
            img = Image.open(self.female_list[idx]).convert('RGB')
            tensor = self.transform(img)
            label = torch.tensor([0.], requires_grad=True)
        
        return tensor, label

        
