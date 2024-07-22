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
    def __init__(self, org_dir, proc_dir, target_size = None, resize = 'resize'):
        self.img_dir = org_dir
        self.label_dir = proc_dir
        self.img_names = np.array(glob.glob1(self.label_dir,'*.npy'))
        # np.random.shuffle(self.img_names)
        self.img_names.tolist()

        original_size = np.load(os.path.join(self.img_dir, self.img_names[0]), allow_pickle = True)[()]['mask_full'].shape
        self.target_size = target_size if target_size != None else original_size
        self.XX, self.YY = torch.meshgrid(torch.linspace(0,1,self.target_size[0],dtype=torch.float32), torch.linspace(0,1,self.target_size[1],dtype=torch.float32))

        if target_size == None:
            self.transform = None 
        elif resize == 'resize':
            self.transform = transforms.Resize(self.target_size, antialias = False)
        elif resize == 'pad':
            self.transform = Pad([(self.target_size[1] - original_size[1])//2, (self.target_size[0] - original_size[0])//2])
            
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.load(img_path, allow_pickle = True)[()]['mask_full']
        img = morphology.remove_small_holes(img>0, area_threshold=10, connectivity=1)

        lab_path = os.path.join(self.label_dir, self.img_names[idx])
        lab = np.load(lab_path, allow_pickle = True)[()]['mask_full']
        lab = morphology.remove_small_holes(lab>0, area_threshold=10, connectivity=1)

        image = torch.from_numpy(img).float()[None,:,:]
        if self.transform != None: image = self.transform(image) 
        image = torch.cat([image,self.XX[None,:,:],self.YY[None,:,:]],dim=0) 
        
        # Getting the cleaning trace
        label = torch.from_numpy(lab).long()[None,:,:]
        if self.transform != None: label = self.transform(label)

        target = {}
        target["image"] = image
        target["mask"] = label
        return target