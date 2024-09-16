""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, up_kernel_size=2, in_dropout_p=0.3, dropout_p=0.1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.atten = nn.Sequential(Attention(in_channel=512, patch_size=1, n_heads=1, n_embd=512, N=32, p_drop=dropout_p),
                                    Attention(in_channel=512, patch_size=1, n_heads=1, n_embd=512, N=32, p_drop=dropout_p))

        
        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))
        # self.atten = BottleneckAttention(512, p_drop=dropout_p)
        self.up1 = (Up(512, 256, up_kernel_size))
        self.up2 = (Up(256, 128, up_kernel_size))
        self.up3 = (Up(128, 64, up_kernel_size))
        self.up4 = (Up(64, 32, up_kernel_size))
        self.outc = (OutConv(32, n_classes))

        self.drop_layer = nn.Dropout(p=dropout_p)
        self.dropout_p = in_dropout_p

    def forward(self, x):
        xd = nn.functional.dropout(x,p=self.dropout_p,training=True)
        # x1 = self.in_att(xd)
        x1 = self.inc(xd)
        x1 = self.drop_layer(x1)
        x2 = self.down1(x1)
        x2 = self.drop_layer(x2)
        x3 = self.down2(x2)
        x3 = self.drop_layer(x3)
        x4 = self.down3(x3)
        x4 = self.drop_layer(x4)
        x5 = self.down4(x4)
        
        x5 = self.atten(x5)
        
        x = self.up1(x5, x4)
        x = self.drop_layer(x)
        x = self.up2(x, x3)
        x = self.drop_layer(x)
        x = self.up3(x, x2)
        x = self.drop_layer(x)
        x = self.up4(x, x1)
        x = self.drop_layer(x)
        logits = self.outc(torch.cat([xd]*32+x, dim=1))
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)