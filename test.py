package_path = "."
import sys
sys.path.append(package_path)
from PIL import Image
import os
import cv2
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model import regressor
import warnings; warnings.filterwarnings('ignore')

from auto_grid import AutoGrid



cwd = os.path.dirname(os.path.realpath(__file__))

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])    

class GenderFilter(nn.Module):
    def __init__(self, ckpt_path = 'ckpt/gender.pt'):
        super(GenderFilter, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gender_filter = regressor().to(self.device).eval()
        ckpt = torch.load(os.path.join(cwd, ckpt_path), map_location=self.device)
        self.gender_filter.load_state_dict(ckpt)
        for param in self.gender_filter.parameters():
            param.requires_grad = False
        del ckpt
        
    def forward(self, tensor):
        tensor = F.interpolate(tensor, (256, 256), mode='bilinear')
        score = self.gender_filter(tensor)
        return score

GF = GenderFilter()
result_path = 'assets/result_images'
os.makedirs(result_path, exist_ok=True)
image_path_list = sorted(glob.glob('./assets/test_images/*.*'))

ag = AutoGrid(x_box_num=4, y_box_num=4)

for idx ,image_path in enumerate(image_path_list):
    target = Image.open(image_path).convert('RGB')
    target_ = transform(target).unsqueeze(0).cuda()
    pred = GF(target_)
    pred = pred.squeeze().detach().cpu().numpy()

    target = np.array(target)
    result = cv2.putText(target, str(np.round(pred, 4)), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 63, 0), 20)
    ag.add(result[:, :, ::-1], idx//4, idx%4, 1, 1)

ag.save("grid.png")
