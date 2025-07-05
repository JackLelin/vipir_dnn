import numpy as np
import torch
from torch.utils.data import Dataset
import glob, os
# from torchvision.io import read_image
# import torchvision
from torchvision import transforms
from torchvision.transforms.v2 import Pad

from skimage import morphology

# from PIL import Image
# from torchvision.transforms import ToTensor

class HandPickTraceDataset(Dataset):
    def __init__(self, label_dir, target_size = None, resize = 'resize'):
        self.label_dir = label_dir
        self.img_names = np.array(glob.glob1(self.label_dir,'*.npy'))
        # np.random.shuffle(self.img_names)
        self.img_names.tolist()

        original_size = np.load(os.path.join(self.label_dir, self.img_names[0]), allow_pickle = True)[()]['mask_full'].shape
        self.target_size = target_size if target_size != None else original_size
        # self.XX, self.YY = torch.meshgrid(torch.linspace(0,1,self.target_size[0],dtype=torch.float32), torch.linspace(0,1,self.target_size[1],dtype=torch.float32))

        if target_size == None:
            self.transform = None 
        elif resize == 'resize':
            self.transform = transforms.Resize(self.target_size, antialias = False)
        elif resize == 'pad':
            self.transform = Pad([(self.target_size[1] - original_size[1])//2, (self.target_size[0] - original_size[0])//2])
            
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        #crop_region:[120:-10,30:390]
        img_path = os.path.join(self.label_dir, self.img_names[idx])
        data = np.load(img_path, allow_pickle = True)[()]
        

        image = torch.from_numpy(data['mask_full']).float()[None,:,:]
        if self.transform != None: image = self.transform(image) 
        # image = torch.cat([image,self.XX[None,:,:],self.YY[None,:,:]],dim=0) 
        
        # Getting the cleaning trace
        imgO = data['otrace']
        imgX = data['xtrace']
        label = np.zeros((2,imgO.shape[0],imgO.shape[1]),dtype=np.int_)

        label[0,:,:][imgO > 0] = 1 # R channel, O mode
        label[1,:,:][imgX > 0] = 1 # G channel, X mode

        label = torch.from_numpy(label).long()
        if self.transform != None: label = self.transform(label)
    
        target = {}
        target["image"] = image
        target["mask"] = label
        return target