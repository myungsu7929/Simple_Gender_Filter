import os, cv2, glob, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image




#####################################
# Dataset
#####################################
class FaceswapDataset(Dataset):
    def __init__(self, image_path_list):
        self.image_path_list = image_path_list

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self,idx):
        image_path  = self.image_path_list[idx]
        image = Image.open(image_path)
        return self.transforms(image), image_path

    def __len__(self):
        return len(self.image_path_list)


#####################################
# Model
#####################################
class regressor(nn.Module):
    def __init__(self, attrib_num):
        super().__init__()
        self.mobilenet_model = mobilenet_v2(pretrained=True).cuda().eval() # pretrained mobilenet_model
        for params in self.mobilenet_model.parameters():
            params.require_grad = False

        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, attrib_num)      # attr 개수 고려하기
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.mobilenet_model(x) # 8 1000
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        return self.act(x) *0.5 + 0.5


myreg = regressor(1).cuda().eval()
ckpt = torch.load("gender.pt")
myreg.load_state_dict(ckpt)

load_roots = [
    "/home/compu/dataset/kface_gender_SR/kface_female/insta",
    "/home/compu/dataset/kface_gender_SR/kface_male/insta"
    ]

save_roots = [
    "/home/compu/dataset/KF-dataset/KFW/KFW_HR/KFW_HR_F/insta",
    "/home/compu/dataset/KF-dataset/KFW/KFW_HR/KFW_HR_M/insta"
]

for save_root in save_roots:
    os.makedirs(save_root, exist_ok=True)

male_count = 0
female_count = 0
for load_root in load_roots:
    img_paths = glob.glob(f"{load_root}/*.*")


    dataset = FaceswapDataset(img_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    dataloader_iter = iter(dataloader)

    for (img, img_path) in tqdm(dataloader_iter):
        img = img.cuda()
        predicts = myreg(img) # 8 6
        # print(predicts)

        for i in range(len(img)):
            if predicts[i] > 0.5:
                save_path = f"{save_roots[1]}/{str(male_count).zfill(8)}.png"
                shutil.copy(img_path[i], save_path)
                male_count += 1

            else:
                save_path = f"{save_roots[0]}/{str(female_count).zfill(8)}.png"
                shutil.copy(img_path[i], save_path)
                female_count += 1

