""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class TransformerNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_size, in_dropout_p=0.3, dropout_p=0.3, up_kernel_size=2):
        super(TransformerNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.att = Transformer(n_heads=1, n_embd=1024, n_blocks=16, p_drop=dropout_p, n_token=input_size[1], n_embd_in=input_size[0], n_channel_out=n_classes)

        self.inc = (DoubleConv(n_classes, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024, 512, up_kernel_size))
        self.up2 = (Up(512, 256, up_kernel_size))
        self.up3 = (Up(256, 128, up_kernel_size))
        self.up4 = (Up(128, 64, up_kernel_size))
        self.outc = (OutConv(64+n_channels, n_classes))

        # self.drop_layer = nn.Dropout(p=dropout_p)
        self.dropout_p = in_dropout_p

    def forward(self, x):
        xd = nn.functional.dropout(x,p=self.dropout_p,training=True)
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

        logits = self.outc(torch.cat([xu,xd], dim=1)) #torch.cat([x]*32, dim=1)+
        return logits

    def use_checkpointing(self):
        self.att = torch.utils.checkpoint(self.att)
        self.outc = torch.utils.checkpoint(self.outc)
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        # self.outc = torch.utils.checkpoint(self.outc)