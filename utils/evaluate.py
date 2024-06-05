import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)

    dice_score = 0
    
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0, leave=True):
        image, label = batch['image'], batch['mask']

        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        mask_pred = net(image)
        imgO = torch.argmax(mask_pred[:,0:2,:,:],dim=1)
        imgX = torch.argmax(mask_pred[:,2:4,:,:],dim=1)

        mask_pred = ((imgO + imgX) > 0).float()[:,None,:,:]

        dice_score += dice_coeff(mask_pred, label, reduce_batch_first=False)
    net.train()
    return dice_score / max(num_val_batches, 1)

#     # iterate over the validation set
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            
#             # move images and labels to correct device and type
#             image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#             mask_true = mask_true.to(device=device, dtype=torch.long)

#             # predict the mask
#             mask_pred = net(image)

#             if net.n_classes == 1:
#                 assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             else:
#                 assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
#                 # convert to one-hot format
# #                 mask_true = F.one_hot(mask_true, net.n_classes).float()
# #                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).float()
#                 # compute the Dice score, ignoring background
#                 dice_score += multiclass_dice_coeff(F.softmax(mask_pred, dim=1).float(), mask_true.float(), reduce_batch_first=False)

#     net.train()
#     return dice_score / max(num_val_batches, 1)
