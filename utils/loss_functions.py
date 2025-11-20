import torch
import torch.nn as nn
import torch.nn.functional as F

class DBCEF(nn.Module):
    """ DBCEF from paper https://arxiv.org/pdf/1906.05229 """

    def __init__(self, epsilon: float = 1e-4, gamma: float = 3.3, nue: float = 1.15):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = epsilon
        self.gamma = gamma
        self.nue = nue

    def forward(self, input : torch.Tensor, target : torch.Tensor):
        y = torch.clamp(self.softmax(input), min = 1e-10, max = 1-1e-10)
        t = nn.functional.one_hot(target, num_classes = 2).permute(0, 3, 1, 2).contiguous()

        N = t.size()[-1]*t.size()[-2]
        beta = torch.sum(t,dim=(-1,-2))/N + self.epsilon 
        beta = beta[:,:,None,None]**self.nue

        CE = t*torch.clamp(torch.log(y), min = -100)
        FCE = (1-y)**self.gamma*CE
        # FCE = CE
        DBCEF = FCE/beta

        return -1*torch.sum(DBCEF)/N/y.size()[0]
    
class BCE_MSE(nn.Module):
    def __init__(self, epsilon: float = 1e-4, beta_bce=1.0):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.epsilon = epsilon
        self.beta_bce = beta_bce

    def forward(self, pred, y_row, y_mask,):
        """
        Combined BCE + masked MSE loss.

        Args:
            pred:     [B, N, 2] network output, [:,:,0] = row, [:,:,1] = existence (logit or prob)
            y_row:    [B, N] ground-truth row indices, 0..M-1
            y_mask:   [B, N] 1 if trace exists, 0 if no trace
            beta_bce: weighting for BCE term
        """
        # Split outputs
        pred_row = torch.sigmoid(pred[:,:,0])  # ensure in [0,1]
        pred_exist = torch.sigmoid(pred[:,:,1])  # existence prob

        # Normalize ground truth row to [0,1]
        y_t = y_row.float() 
        mask = y_mask.float()

        # --- Masked MSE loss (only where mask==1) ---
        num_valid = mask.sum().clamp(min=1.0)
        mse_loss = ((pred_row - y_t) ** 2 * mask).sum() / num_valid

        # --- BCE loss (all columns) ---
        bce_loss = F.binary_cross_entropy(pred_exist, mask)

        total_loss = mse_loss + self.beta_bce * bce_loss

        return total_loss, mse_loss.item(), bce_loss.item()