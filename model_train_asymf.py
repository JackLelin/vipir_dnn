import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.v2 import Pad
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import glob

from utils.evaluate import evaluate
from unet import UNet, TransformerNet, TransformerAsymf

import numpy as np
import matplotlib.pyplot as plt

from fakeionogramdatasetOX import ParabolaIonoDatasetOX
from testionogramdataset import HandPickTraceDataset

from torch.optim import AdamW

class FOFXloss(nn.Module):
    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.sigmoid = nn.Sigmoid()

    def forward(self, input : torch.Tensor, target : torch.Tensor):
        exist = torch.clamp(self.sigmoid(input[:,0:4]), min = 1e-10, max = 1-1e-10) 
        loc = input[:,4:]
        # print(exist)
        loss = 0.0
        loss += -1*torch.sum(target[:,0:4] * torch.clamp(torch.log(exist), min = -100, max = 0))# +(1-target[:,0:4]) * torch.clamp(torch.log(1-exist), min = -100))
        # print(loss)
        loss += torch.sum(target[:,0:4] * (loc - target[:,4:])**2)
        # print(loss)
        return loss/input.size()[0]

@torch.no_grad()
def random_asymf_test(model, epoch_num):    
    with torch.inference_mode():
        img = []
        filename = []
        for _ in range(5):
            file = testfiles[np.random.randint(0,len(testfiles)-1)]
            filename += [file]
            testdata = np.load('VIPIR_PROC/proc/' + file, allow_pickle=True, fix_imports=True)[()]
            testimage = torch.from_numpy(testdata['mask_full']).float()
            testimage = testimage[None,None,:,:] if resize_tr == None else resize_tr(testimage[None,None,:,:])
            # testimage = torch.cat([testimage,XX[None,None,:,:],YY[None,None,:,:]],dim=-3)
            img += [testimage]
        
        img = torch.cat(img,dim=0)
        img = img.to('cuda')
        label = model(img)
        sigmoid = nn.Sigmoid()
        for i in range(5):
            img1,label1 = img[i],label[i]
            label1 = label1.squeeze().to('cpu')

            image = img1.detach().cpu().numpy()[0]
            exist = label1[0:4].detach().numpy()
            locs = label1[4:].detach().numpy().clip(min=0,max=407).astype(np.int_)

            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 5))
            plt.title(f'epoch{epoch_num}_{sigmoid(torch.from_numpy(exist))}')
            ax = axs[0]
            fo = np.zeros_like(image)
            fo[:,[locs[0],locs[1]]] = 1
            ax.imshow(image + fo, origin='lower', aspect = 'auto')
            ax.axis('off')
            ax = axs[1]
            fx = np.zeros_like(image)
            fx[:,[locs[2],locs[3]]] = 1
            ax.imshow(image + fx, origin='lower', aspect = 'auto')
            ax.axis('off')

            plt.savefig(f'testing_asymf_result/epoch{epoch_num}_{filename[i][:-4]}.png', bbox_inches='tight')
            plt.close()
        del label; del img
    return
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

epochs = 15
batch_size = 5
learning_rate = 1e-5
val_percent = 0.1

weight_decay = 5e-3
momentum = 0.2
gradient_clipping = 1.0

target_size = [512,408] #[572,572] #[524,412]
original_size = [512,408]
resize = 'pad'

train_dataset = ParabolaIonoDatasetOX('training_data', target_size= target_size, resize=resize)

n_val = int(len(train_dataset) * val_percent)
n_train = len(train_dataset) - n_val
train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator())

# 3. Create data loaders
loader_args = dict(batch_size=batch_size,  pin_memory=True) #os.cpu_count()num_workers=5,

train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)

test_dataset = HandPickTraceDataset('VIPIR_PROC/proc', 'VIPIR_PROC/dog_pick', target_size= target_size, resize=resize)
test_loader = DataLoader(test_dataset, shuffle=True, drop_last=True, **loader_args)

testfiles = glob.glob1('VIPIR_PROC/proc/','*.npy')

if resize == 'resize':
    resize_tr = transforms.Resize(target_size, antialias = False)
elif resize == 'pad':
    resize_tr = Pad([(target_size[1] - original_size[1])//2, (target_size[0] - original_size[0])//2])
else: resize = None

# XX, YY = torch.meshgrid(torch.linspace(0,1,target_size[0],dtype=torch.float32),torch.linspace(0,1,target_size[1],dtype=torch.float32))

model = TransformerAsymf(n_channels=1, n_classes=1, input_size=target_size, in_dropout_p = 0.25, dropout_p=0.3)
model = model.to(device=device)
# model.load_state_dict(torch.load('saved_model/'+'TOPHAT_checkpoint_epoch_10_18000.pth')['model_state_dict'])
model.train()

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
fofxloss = FOFXloss()
fofxloss = fofxloss.to(device=device)

epcoh_elapsed = 0
losses = []
val_losses = []
test_losses = []

checkpoint_N = 1000
for epoch in range(epcoh_elapsed,epcoh_elapsed+epochs):
    with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='img') as pbar:
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, asymf = data['image'], data['asymf']
            inputs, asymf = inputs.to(device), asymf.to(device) 
            outputs = model(inputs)
            # print(outputs.size())
            loss = fofxloss(outputs, asymf)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.update(inputs.shape[0])
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            del inputs; del asymf
            optimizer.zero_grad()
            if (i+1)%checkpoint_N == 0: # validation for every checkpoint_N batch
                losses += [running_loss / i]
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for ii, data in enumerate(val_loader, 0):
                        inputs, asymf = data['image'], data['asymf']
                        inputs, asymf = inputs.to(device), asymf.to(device) 
                        outputs = model(inputs)
                        loss = fofxloss(outputs, asymf)
                        val_loss += loss.item()
                        del inputs; del asymf
                    avg_val_loss = val_loss / len(val_loader)
                    val_losses += [avg_val_loss]
                    # test_loss = evaluate(model,test_loader,device)
                    # test_losses += [test_loss.cpu()]
                    # print(f'Epoch: {epoch + 1}, Img:{(i+1)*batch_size} ,Validation Loss: {avg_val_loss:.4f}, Testing Dice_score: {test_loss:.4f}')
                    # pick 5 random ionogram for testing
                    random_asymf_test(model, epoch+1)
                model.train()
                    
                fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
                ax = axs[0]
                ax.plot(np.arange(len(losses))*checkpoint_N,losses)
                ax.set_xlabel('Steps');ax.set_ylabel('Training loss')
                ax = axs[1]
                ax.plot(np.arange(len(val_losses))*checkpoint_N,val_losses)
                ax.set_xlabel('Steps');ax.set_ylabel('Val loss')
                # ax = axs[2]
                # ax.plot(np.arange(len(test_losses))*checkpoint_N,test_losses)
                # ax.set_xlabel('Steps');ax.set_ylabel('Val loss')
                fig.savefig(f'losses.png', bbox_inches='tight')
                plt.close()
                
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'saved_model_asymf/checkpoint_epoch_{epoch + 1}_{(i+1)*5}.pth')
            # break
                
        print('-' * 20)
        epcoh_elapsed += 1