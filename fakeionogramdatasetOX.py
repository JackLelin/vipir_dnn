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
        self.asymf_dir = img_dir+'/asym_f/'
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
        image = torch.from_numpy((img[:,:]>128).astype(np.float32)).float()[None,:,:]
        image = torch.flip(self.transform(image),dims=[1])
        # Getting the groud truth label
        imgO = np.array(Image.open(self.gt_dirO + self.img_names[idx]),dtype=np.double)
        imgX = np.array(Image.open(self.gt_dirX + self.img_names[idx]),dtype=np.double)
        label = np.zeros((2,imgO.shape[0],imgO.shape[1]),dtype=np.int_)
        label[0,:,:][imgO[:,:,0] > 128] = 1 # R channel, O mode
        label[1,:,:][imgX[:,:,1] > 128] = 1 # G channel, X mode
        # Getting fof2
        # asymf = np.load(self.asymf_dir + self.img_names[idx] + '.npy', allow_pickle=True)[()]

        label = torch.from_numpy(label).long()
        label = torch.flip(self.transform(label),dims=[1])
        target = {}
        target["image"] = image
        target["mask"] = label.squeeze()
        # target["asymfO"] = asymf['asym_fO'][-1]
        # target["asymfX"] = asymf['asym_fX']
        return target