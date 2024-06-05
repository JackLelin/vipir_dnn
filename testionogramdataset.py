import numpy as np
import torch
from torch.utils.data import Dataset
import glob, os
from torchvision.io import read_image
# import torchvision
from torchvision import transforms

from PIL import Image
# from torchvision.transforms import ToTensor

class ParabolaIonoDatasetOX(Dataset):
    def __init__(self, org_dir, proc_dir):
        self.img_dir = org_dir
        self.label_dir = proc_dir
        self.transform = transforms.Resize((572, 572))

        self.img_names = np.array(glob.glob1(self.label_dir,'*.png'))
        np.random.shuffle(self.img_names)
        self.img_names.tolist()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.load(img_path, allow_pickle = True)[()]['mask_full']
        lab_path = os.path.join(self.label_dir, self.img_names[idx])
        lab = np.load(lab_path, allow_pickle = True)[()]['mask_full']

        image = torch.from_numpy(img).float()[None,:,:]
        image = torch.flip(self.transform(image),dims=[1])
        # Getting the cleaning trace
        label = torch.from_numpy(label).long()[None,:,:]
        image = torch.flip(self.transform(image),dims=[1])

        target = {}
        target["image"] = image
        target["mask"] = label
        return target