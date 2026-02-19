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

    def forward(self, pred, ground_t):
        """
        Combined BCE + masked MSE loss.

        Args:
            pred:     [B, N, 2] network output, [:,:,0] = row, [:,:,1] = existence (logit or prob)
            y_row:    [B, N] ground-truth row indices, 0..M-1
            y_mask:   [B, N] 1 if trace exists, 0 if no trace
            beta_bce: weighting for BCE term
        """
        # Split outputs
        pred_pres_o = torch.sigmoid(pred[:,0,:])  # ensure in [0,1]
        pred_hidx_o = torch.sigmoid(pred[:,1,:])  # existence prob
        pred_pres_x = torch.sigmoid(pred[:,2,:])  # ensure in [0,1]
        pred_hidx_x = torch.sigmoid(pred[:,3,:])  # existence prob

        # Normalize ground truth row to [0,1]
        mask_o = ground_t[:,0,:]
        y_t_o = ground_t[:,1,:]
        mask_x = ground_t[:,2,:]
        y_t_x = ground_t[:,3,:]

        # --- Masked MSE loss (only where mask==1) ---
        num_valid_o = mask_o.sum().clamp(min=1.0)
        mse_loss_o = ((pred_hidx_o - y_t_o) ** 2 * mask_o).sum() / num_valid_o
        num_valid_x = mask_x.sum().clamp(min=1.0)
        mse_loss_x = ((pred_hidx_x - y_t_x) ** 2 * mask_x).sum() / num_valid_x

        # --- BCE loss (all columns) ---
        bce_loss_o = F.binary_cross_entropy(pred_pres_o, mask_o)
        bce_loss_x = F.binary_cross_entropy(pred_pres_x, mask_x)

        total_loss = mse_loss_o + self.beta_bce * bce_loss_o + mse_loss_x + self.beta_bce * bce_loss_x

        return total_loss
    

class CE_by_col(nn.Module):
    def __init__(self, epsilon: float = 1e-9):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.epsilon = epsilon

    def forward(self, pred, ground_t):
        assert pred.size() == ground_t.size()
        
        y = torch.clamp(self.softmax(pred), min = self.epsilon, max = 1-self.epsilon)
        N = ground_t.size()[-1]*ground_t.size()[-2]

        CE = -1*512* ground_t*torch.clamp(torch.log(y), min = -100)

        total_loss = torch.sum(CE)/N/y.size()[0]
        return total_loss