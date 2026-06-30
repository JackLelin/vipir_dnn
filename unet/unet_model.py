""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    '''
    Docstring for UNet
    To minimize the amount of memory operation, the UNet is taking transposed image as input, 
    i.e., [B, C, W, H] where W is the freqs axis (number of columns), H is the height axis (number of rows),
    in other words, column major format.
    '''
    def __init__(self, n_channels, n_classes, input_size, in_dropout_p=0.3, dropout_p=0.1, up_kernel_size=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Transformer-only model: we output directly to the required number of classes
        self.att = Transformer(n_heads=4, dim_embd=1024, n_blocks=16, p_drop=dropout_p, n_token=input_size[0], dim_embd_in=input_size[1], n_channel_out=n_classes)
        
        self.dropout_p = in_dropout_p

    def forward(self, x):
        xd = nn.functional.dropout(x, p=self.dropout_p, training=True)
        xa = self.att(xd)
        return xa