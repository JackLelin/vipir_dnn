import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.v2 import Pad
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import glob

from utils.evaluate import evaluate
from unet import UNet, TransformerNet

import numpy as np
import matplotlib.pyplot as plt

from fakeionogramdatasetOX import ParabolaIonoDatasetOX
from testionogramdataset import HandPickTraceDataset

from torch.optim import AdamW

class DBCEF(nn.Module):
    """ DBCEF from paper https://arxiv.org/pdf/1906.05229 """

    def __init__(self, epsilon: float = 1e-4, gamma: float = 1):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, input : torch.Tensor, target : torch.Tensor):
        y = torch.clamp(self.softmax(input), min = 1e-10, max = 1-1e-10)
        t = nn.functional.one_hot(target, num_classes = 2).permute(0, 3, 1, 2).contiguous()

        N = t.size()[-1]*t.size()[-2]
        beta = torch.sum(t,dim=(-1,-2))/N + self.epsilon 
        beta = beta[:,:,None,None]

        CE = t*torch.clamp(torch.log(y), min = -100)
        FCE = (1-y)**self.gamma*CE
        # FCE = CE
        DBCEF = FCE/beta

        return -1*torch.sum(DBCEF)/N/y.size()[0]

@torch.no_grad()
def test_on_TOPHAT(model, epoch_num):    
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
        for i in range(5):
            img1,label1 = img[i],label[i]
            label1 = label1.squeeze().to('cpu')
            imgO = torch.argmax(label1[0:2,:,:],dim=0).detach().numpy()
            imgX = torch.argmax(label1[2:4,:,:],dim=0).detach().numpy()
            fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
            plt.title(f'epoch{epoch_num}_{filename[i]}')
            ax = axs[0]
            ax.imshow(img1.detach().cpu().numpy()[0], origin='lower', aspect = 'auto')
            ax.axis('off')
            ax = axs[1]
            ax.imshow(imgO, origin='lower', aspect = 'auto')
            ax.axis('off')
            ax = axs[2]
            ax.imshow(imgX, origin='lower', aspect = 'auto')
            ax.axis('off')
            ax = axs[3]
            imgX[imgO==1]=2
            ax.imshow(imgX, origin='lower', aspect = 'auto')
            ax.axis('off')

            plt.savefig(f'testing_result/epoch{epoch_num}_{filename[i][:-4]}.png', bbox_inches='tight')
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

model = TransformerNet(n_channels=1, n_classes=4, input_size=target_size, in_dropout_p = 0.25, dropout_p=0.3)
# model = UNet(n_channels=1, n_classes=4, in_dropout_p = 0.33, dropout_p=0.25)
model = model.to(device=device)
# model.load_state_dict(torch.load('saved_model/'+'TOPHAT_checkpoint_epoch_10_18000.pth')['model_state_dict'])
model.train()

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
dbcefloss = DBCEF()
dbcefloss = dbcefloss.to(device=device)

epcoh_elapsed = 0
losses = []
val_losses = []
test_losses = []

checkpoint_N = 1000
for epoch in range(epcoh_elapsed,epcoh_elapsed+epochs):
    with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}', unit='img') as pbar:
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['image'], data['mask']
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            loss = dbcefloss(outputs[:,0:2,:,:], labels[:,0,:,:]) + dbcefloss(outputs[:,2:4,:,:], labels[:,1,:,:])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.update(inputs.shape[0])
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            del inputs; del labels
            optimizer.zero_grad()
            if (i+1)%checkpoint_N == 0: # validation for every checkpoint_N batch
                losses += [running_loss / i]
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for ii, data in enumerate(val_loader, 0):
                        inputs, labels = data['image'], data['mask']
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)                
                        loss = dbcefloss(outputs[:,0:2,:,:], labels[:,0,:,:]) + dbcefloss(outputs[:,2:4,:,:], labels[:,1,:,:])
                        val_loss += loss.item()
                        del inputs; del labels
                    avg_val_loss = val_loss / len(val_loader)
                    val_losses += [avg_val_loss]
                    test_loss = evaluate(model,test_loader,device)
                    test_losses += [test_loss.cpu()]
                    print(f'Epoch: {epoch + 1}, Img:{(i+1)*batch_size} ,Validation Loss: {avg_val_loss:.4f}, Testing Dice_score: {test_loss:.4f}')
                    # pick 5 random ionogram for testing
                    test_on_TOPHAT(model, epoch+1)
                model.train()
                    
                fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
                ax = axs[0]
                ax.plot(np.arange(len(losses))*checkpoint_N,losses)
                ax.set_xlabel('Steps');ax.set_ylabel('Training loss')
                ax = axs[1]
                ax.plot(np.arange(len(val_losses))*checkpoint_N,val_losses)
                ax.set_xlabel('Steps');ax.set_ylabel('Val loss')
                ax = axs[2]
                ax.plot(np.arange(len(test_losses))*checkpoint_N,test_losses)
                ax.set_xlabel('Steps');ax.set_ylabel('Val loss')
                fig.savefig(f'losses.png', bbox_inches='tight')
                plt.close()
                
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f'saved_model/TOPHAT_checkpoint_epoch_{epoch + 1}_{(i+1)*5}.pth')
            # break
                
        print('-' * 20)
        epcoh_elapsed += 1