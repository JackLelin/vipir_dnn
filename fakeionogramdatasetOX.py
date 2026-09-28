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
        self.npy_dir = os.path.join(img_dir, 'npy')

        self.img_names = sorted(glob.glob1(self.npy_dir, '*.npy'))

        sample_data = np.load(os.path.join(self.npy_dir, self.img_names[0]), allow_pickle=True)[()]
        original_size = sample_data['train'].T.shape

        self.target_size = target_size if target_size != None else original_size
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
        file_path = os.path.join(self.npy_dir, self.img_names[idx])
        data = np.load(file_path, allow_pickle=True)[()]

        train_img = data['train'].astype(np.float32)
        omode_gaussian = data['omode_mask'].astype(np.float32)
        xmode_gaussian = data['xmode_mask'].astype(np.float32)

        image = torch.from_numpy(train_img).float()[None, :, :]
        image = torch.permute(image, (0, 2, 1)).contiguous()
        if self.transform != None:
            image = self.transform(image)

        target_o = torch.from_numpy(omode_gaussian).float()[None, :, :]
        target_x = torch.from_numpy(xmode_gaussian).float()[None, :, :]
        
        target_o = torch.permute(target_o, (0, 2, 1)).contiguous()
        target_x = torch.permute(target_x, (0, 2, 1)).contiguous()

        # random crop and resize on both image and labels
        cropped = self.cropresize(torch.cat([image, target_o, target_x], dim=0))
        image = cropped[0:1, :, :]
        target_o = cropped[1:2, :, :]
        target_x = cropped[2:3, :, :]

        target = {}
        target["image"] = image
        target["gt"] = torch.cat([target_o, target_x], dim=0)

        return target

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # Set num_workers to 0 for a quick local test
    loader_args = dict(batch_size=1,  pin_memory=True, num_workers=0)
    # Initialize with a larger sigma so we can see the effect clearly
    train_dataset = ParabolaIonoDatasetOX('training_data', target_size=[408, 512], resize='pad')
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)

    data = next(iter(train_loader))
    print(data['image'].size())
    print(data['gt'].size())
    
    # data['image'] is [1, 1, H, W], data['gt'] is [1, 2, H, W]
    image = data['image'][0, 0].numpy()
    gt_o = data['gt'][0, 0].numpy()
    gt_x = data['gt'][0, 1].numpy()

    print(f"Image shape: {image.shape}, GT shape: {data['gt'].shape}")

    # Plot the results
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
    
    axs[0].imshow(image.T, origin='lower', aspect='auto', cmap='gray', interpolation='none')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    axs[1].imshow(gt_o.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
    axs[1].set_title('GT O-mode (EDT + Gaussian)')
    axs[1].axis('off')
    
    axs[2].imshow(gt_x.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
    axs[2].set_title('GT X-mode (EDT + Gaussian)')
    axs[2].axis('off')
    
    # Overlay: Image + GT
    axs[3].imshow(image.T, origin='lower', aspect='auto', cmap='gray', interpolation='none')
    # Overlay the combined heatmap with some transparency (alpha=0.6)
    axs[3].imshow((gt_o + gt_x).T, origin='lower', aspect='auto', cmap='magma', alpha=0.6, interpolation='none')
    axs[3].set_title('Overlay (Image + Combined GT)')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()
