import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
# import torchvision
from torchvision import transforms

from PIL import Image
# from torchvision.transforms import ToTensor

class ParabolaIonoDatasetOX(Dataset):
    def __init__(self, img_dir, target_transform=None):
        self.img_dir = img_dir+'/train'
        self.gt_dirO = img_dir+'/O/'
        self.gt_dirX = img_dir+'/X/'
        self.transform = transforms.Resize((572, 572))
        self.target_transform = target_transform
        self.img_names = np.array(os.listdir(self.img_dir))
        np.random.shuffle(self.img_names)
        self.img_names.tolist()
#         self.totensor = ToTensor()
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.array(Image.open(img_path),dtype=np.double)
        image = torch.from_numpy((np.mean(img[:,:,0:3],axis=2)<200).astype(np.float32)).float()[None,:,:]
        image = torch.flip(self.transform(image),dims=[1])
#         image = image.repeat(3, 1, 1)
        # Getting the groud truth label
        imgO = np.array(Image.open(self.gt_dirO + self.img_names[idx]),dtype=np.double)
        imgX = np.array(Image.open(self.gt_dirX + self.img_names[idx]),dtype=np.double)
        label = np.zeros((2,imgO.shape[0],imgO.shape[1]),dtype=np.int_)
        label[1,:,:][(imgX[:,:,0] - np.mean(imgX[:,:,0:3],axis = 2)) > 0] = 1 # R channel, X mode
        label[0,:,:][(imgO[:,:,2] - np.mean(imgO[:,:,0:3],axis = 2)) > 0] = 1 # B channel, O mode
        # overlap area will be 3
        label = torch.from_numpy(label).long()
        label = torch.flip(self.transform(label),dims=[1])
        target = {}
        target["image"] = image
        target["mask"] = label.squeeze()
        
        return target