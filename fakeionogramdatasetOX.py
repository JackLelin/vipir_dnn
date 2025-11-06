import numpy as np
import torch
from torch.utils.data import Dataset
import glob, os
from torchvision.io import read_image
# import torchvision
from torchvision import transforms
from torchvision.transforms.v2 import Pad

from PIL import Image
# from torchvision.transforms import ToTensor

class ParabolaIonoDatasetOX(Dataset):
    def __init__(self, img_dir, target_size = None, resize = 'resize'):
        self.img_dir = img_dir+'/train'
        self.gt_dirO = img_dir+'/O/'
        self.gt_dirX = img_dir+'/X/'
        self.asymf_dir = img_dir+'/asym_f/'
        self.img_names = np.array(glob.glob1(self.img_dir,'*.png'))
        self.vipir_freqs = np.load('JM91J_2016014141141.npy', allow_pickle=True)[()]['vipir_freqs']
        # np.random.shuffle(self.img_names)
        self.img_names.sort()
        self.img_names.tolist()

        original_size = np.array(Image.open(os.path.join(self.img_dir, self.img_names[0]))).shape
        self.target_size = target_size if target_size != None else original_size
        # self.XX, self.YY = torch.meshgrid(torch.linspace(0,1,self.target_size[0],dtype=torch.float32), torch.linspace(0,1,self.target_size[1],dtype=torch.float32))
        if target_size == None:
            self.transform = None 
        elif resize == 'resize':
            self.transform = transforms.Resize(self.target_size, antialias = False)
        elif resize == 'pad':
            self.transform = Pad([(self.target_size[1] - original_size[1])//2, (self.target_size[0] - original_size[0])//2])
        
        self.cropresize = transforms.RandomResizedCrop(self.target_size, scale=(0.9, 1.1), ratio=(0.8, 1.2), interpolation=transforms.InterpolationMode.NEAREST)
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        #crop_region:[120:-10,30:390]
        img_path = os.path.join(self.img_dir, self.img_names[idx]) # type: ignore
        img = np.array(Image.open(img_path))
        image = torch.from_numpy((img>128).astype(np.float32)).float()[None,:,:]
        if self.transform != None: image = self.transform(image)
        image = torch.flip(image,dims=[1])
        
        # Getting the groud truth label
        imgO = np.array(Image.open(self.gt_dirO + self.img_names[idx]))
        imgX = np.array(Image.open(self.gt_dirX + self.img_names[idx]))
        label = np.zeros((2,imgO.shape[0],imgO.shape[1]),dtype=np.int_)
        
        label[0,:,:][imgO[:,:,0] > 128] = 1 # R channel, O mode
        label[1,:,:][imgX[:,:,1] > 128] = 1 # G channel, X mode

        label = torch.from_numpy(label)
        if self.transform != None: label = self.transform(label)
        label = torch.flip(label,dims=[-2])

        # random crop and resize on both image and labels
        cropped = self.cropresize(torch.cat([image,label],dim=0))
        image = cropped[:1,:,:]
        label = cropped[1:,:,:].long()
        # image = torch.cat([image,self.XX[None,:,:],self.YY[None,:,:]],dim=0)
        target = {}
        target["image"] = image
        target["mask"] = label.squeeze()
        target["filename"] = idx
        # Getting fof2
        # asymf = np.load(self.asymf_dir + self.img_names[idx] + '.npy', allow_pickle=True)[()]
        # asymf_tensor = torch.zeros(8)
        # if np.sum(imgO[:,:,0] > 128) > 0:
        #     asymf_tensor[0] = 1 if len(asymf['asym_fO'])>1 else 0 # foE
        #     asymf_tensor[2] = 1 if len(asymf['asym_fX'])>1 else 0 # fxE

        #     asymf_tensor[4] = np.argmin(np.abs(self.vipir_freqs - asymf['asym_fO'][0]))
        #     asymf_tensor[5] = np.argmin(np.abs(self.vipir_freqs - asymf['asym_fO'][-1]))
        #     asymf_tensor[6] = np.argmin(np.abs(self.vipir_freqs - asymf['asym_fX'][0]))
        #     asymf_tensor[7] = np.argmin(np.abs(self.vipir_freqs - asymf['asym_fX'][-1]))
        # target["asymf"] = asymf_tensor
        return target