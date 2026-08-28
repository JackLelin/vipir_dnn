import numpy as np
import torch
from torch.utils.data import Dataset
import glob, os
from torchvision.io import read_image
# import torchvision
from torchvision import transforms
from torchvision.transforms.v2 import Pad

from PIL import Image
import torch.nn.functional as F
# from torchvision.transforms import ToTensor

class ParabolaIonoDatasetOX(Dataset):
    def __init__(self, img_dir, target_size = None, resize = 'resize'):
        self.img_dir = img_dir+'/train'
        # self.gt_dirO = img_dir+'/O/'
        # self.gt_dirX = img_dir+'/X/'
        self.gt_dir = img_dir+'/npy/'
        self.img_names = np.array(glob.glob1(self.img_dir,'*.png'))
        self.vipir_freqs = np.load('JM91J_2016014141141.npy', allow_pickle=True)[()]['vipir_freqs']
        # np.random.shuffle(self.img_names)
        self.img_names.sort()
        self.img_names.tolist()

        original_size = np.array(Image.open(os.path.join(self.img_dir, self.img_names[0]))).T.shape
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
        # print('numpy shape',img.shape)
        image = torch.from_numpy((img>128).astype(np.float32)).float()[None,:,:]
        image = torch.permute(image, (0,2,1))
        image = torch.flip(image,dims=[-1]).contiguous()
        # print('shape after flip',image.shape)
        if self.transform != None: image = self.transform(image)
        # print('shape after transform',image.shape)

        # target["filename"] = idx

        data = np.load(self.gt_dir + self.img_names[idx] + '.npy', allow_pickle=True)[()]

        target_o = torch.from_numpy(data['omode_mask']).float()[None,:,:]
        target_x = torch.from_numpy(data['xmode_mask']).float()[None,:,:]
        
        target_o = torch.permute(target_o, (0,2,1))
        target_x = torch.permute(target_x, (0,2,1))

        # random crop and resize on both image and labels
        cropped = self.cropresize(torch.cat([image, target_o, target_x], dim=0))
        image = cropped[0:1,:,:]
        target_o = cropped[1:2,:,:]
        target_x = cropped[2:3,:,:]

        target = {}
        target["image"] = image
        target["gt"] = torch.cat([target_o, target_x], dim=0)

        return target

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    loader_args = dict(batch_size=1,  pin_memory=True, num_workers=10)
    train_dataset = ParabolaIonoDatasetOX('training_data', target_size= [512,408], resize='pad')
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)

    data = next(iter(train_loader))
    print(data['image'].size())
    print(data['gt'].size())
