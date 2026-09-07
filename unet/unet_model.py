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
        n_att_channel = 4
        # Transformer-only model: we output directly to the required number of classes
        self.att = Transformer(n_heads=4, dim_embd=1024, n_blocks=16, p_drop=dropout_p, n_token=input_size[0], dim_embd_in=input_size[1], n_channel_out=n_att_channel)
        
        self.dropout_p = in_dropout_p

        self.inc = (DoubleConv(n_att_channel, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512, up_kernel_size))
        self.up2 = (Up(512, 256, up_kernel_size))
        self.up3 = (Up(256, 128, up_kernel_size))
        self.up4 = (Up(128, 64, up_kernel_size))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        xd = nn.functional.dropout(x, p=self.dropout_p, training=True)
        xa = self.att(xd)

        x1 = self.inc(xa)
        # x1 = self.drop_layer(x1)
        x2 = self.down1(x1)
        # x2 = self.drop_layer(x2)
        x3 = self.down2(x2)
        # x3 = self.drop_layer(x3)
        x4 = self.down3(x3)
        # x4 = self.drop_layer(x4)
        x5 = self.down4(x4)
        
        xu = self.up1(x5, x4)
        # xu = self.drop_layer(xu)
        xu = self.up2(xu, x3)
        # xu = self.drop_layer(xu)
        xu = self.up3(xu, x2)
        # xu = self.drop_layer(xu)
        xu = self.up4(xu, x1)
        # xu = self.drop_layer(xu)
        logits = self.outc(torch.cat([xd]*64 , dim=1) + xu) #

        return logits