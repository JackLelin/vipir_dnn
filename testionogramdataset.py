import numpy as np
import torch
from torch.utils.data import Dataset
import glob, os
# from torchvision.io import read_image
# import torchvision
from torchvision import transforms

from skimage import morphology

# from PIL import Image
# from torchvision.transforms import ToTensor

class HandPickTraceDataset(Dataset):
    def __init__(self, org_dir, proc_dir):
        self.img_dir = org_dir
        self.label_dir = proc_dir
        # self.transform = transforms.Resize((572, 572), antialias = False)
        self.img_names = np.array(glob.glob1(self.label_dir,'*.npy'))
        img = np.load(os.path.join(self.img_dir, self.img_names[0]), allow_pickle = True)[()]['mask_full']
        self.XX, self.YY = torch.meshgrid(torch.linspace(0,1,img.shape[0],dtype=torch.float32), torch.linspace(0,1,img.shape[1],dtype=torch.float32))
        # np.random.shuffle(self.img_names)
        self.img_names.tolist()

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
        # image = self.transform(image) 
        image = torch.cat([image,self.XX[None,:,:],self.YY[None,:,:]],dim=0) 
        
        # Getting the cleaning trace
        label = torch.from_numpy(lab).long()[None,:,:]
        # label = self.transform(label)

        target = {}
        target["image"] = image
        target["mask"] = label
        return target