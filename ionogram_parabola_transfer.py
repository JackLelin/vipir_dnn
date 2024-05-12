import os
import numpy as np
import torch
import torch.utils.data

import torchvision.transforms.functional as TF

# from PIL import Image


class IonoDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        # load all image files, sorting them to
        # ensure that they are aligned
        train_npy = np.load(file, allow_pickle=True, fix_imports=True)[()]
    
        self.all_labels = train_npy['all_labels']
        self.all_images = train_npy['all_images']
        
#         self.resize = TF.resize([377,377])
    def __getitem__(self, idx):
        
        img = np.zeros([572,572], dtype=np.float32) #572
        img[:512,:408] = self.all_images[idx]
        img = torch.from_numpy(img) 
#         img = img.unsqueeze(img, 2)
        img = img[None,:,:]
        
        masks = self.all_labels[idx]
        # masks_switch_OX = np.zeros_like(masks)
        # masks_switch_OX[masks == 1] = 2
        # masks_switch_OX[masks == 2] = 1
        
        masks_large = np.zeros([2,572,572]) #572
        masks_large[0,:512,:408][masks == 1] = 1
        masks_large[1,:512,:408][masks == 2] = 1
        masks_large = torch.as_tensor(masks_large, dtype=torch.long)

        target = {}
        target["image"] = img
        target["mask"] = masks_large

        return target

    def __len__(self):
        return len(self.all_images)