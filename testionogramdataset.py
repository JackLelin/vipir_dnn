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

        original_size = np.load(os.path.join(self.label_dir, self.img_names[0]), allow_pickle = True)[()]['mask_full'].T.shape
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
        image = torch.permute(image, (0,2,1)).contiguous()
        if self.transform != None: image = self.transform(image) 
        # image = torch.cat([image,self.XX[None,:,:],self.YY[None,:,:]],dim=0) 
        
        # Getting the cleaning trace
        imgO = data['otrace']
        imgX = data['xtrace']
        label = np.zeros((2,imgO.shape[1],imgO.shape[0]),dtype=np.int_)

        label[0,:,:][imgO.T > 0] = 1 # R channel, O mode
        label[1,:,:][imgX.T > 0] = 1 # G channel, X mode

        label = torch.from_numpy(label).long()
        if self.transform != None: label = self.transform(label)
    
        target = {}
        target["image"] = image
        target["mask"] = label
        return target

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # Set num_workers to 0 for a quick local test
    loader_args = dict(batch_size=1,  pin_memory=True, num_workers=0)
    test_dataset = HandPickTraceDataset('VIPIR_PROC/labeled', target_size=[408, 512], resize='pad')
    test_loader = DataLoader(test_dataset, shuffle=True, drop_last=True, **loader_args)

    data = next(iter(test_loader))
    print(data['image'].size())
    print(data['mask'].size())
    
    # data['image'] is [1, 1, H, W], data['mask'] is [1, 2, H, W]
    image = data['image'][0, 0].numpy()
    gt_o = data['mask'][0, 0].numpy()
    gt_x = data['mask'][0, 1].numpy()

    print(f"Image shape: {image.shape}, GT shape: {data['mask'].shape}")

    # Plot the results
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
    
    axs[0].imshow(image.T, origin='lower', aspect='auto', cmap='gray', interpolation='none')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    axs[1].imshow(gt_o.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
    axs[1].set_title('GT O-mode')
    axs[1].axis('off')
    
    axs[2].imshow(gt_x.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
    axs[2].set_title('GT X-mode')
    axs[2].axis('off')
    
    # Overlay: Image + GT
    axs[3].imshow(image.T, origin='lower', aspect='auto', cmap='gray', interpolation='none')
    # Overlay the combined heatmap with some transparency (alpha=0.6)
    axs[3].imshow((gt_o + gt_x).T, origin='lower', aspect='auto', cmap='magma', alpha=0.6, interpolation='none')
    axs[3].set_title('Overlay (Image + Combined GT)')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()

