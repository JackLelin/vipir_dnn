""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class TransformerAsymf(nn.Module):
    def __init__(self, n_channels, input_size, in_dropout_p=0.3, dropout_p=0.3, up_kernel_size=2):
        super(TransformerAsymf, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes

        self.att = Transformer(n_heads=1, n_embd=1024, n_blocks=16, p_drop=dropout_p, n_token=input_size[1], n_embd_in=input_size[0], n_channel_out=2)
        
        self.inc = (DoubleConv(2, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))
        self.up1 = (Up(512, 256, up_kernel_size))
        self.up2 = (Up(256, 128, up_kernel_size))
        self.up3 = (Up(128, 64, up_kernel_size))
        self.up4 = (Up(64, 32, up_kernel_size))
        self.outc = (OutConv(32+n_channels, 8))

        self.linear1 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(input_size[0], 1))
        self.linear2 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(input_size[1], 1))
        # self.softmax = nn.Softmax(dim=1)
        self.dropout_p = in_dropout_p

    def forward(self, x: torch.Tensor):
        B,C,Nr,Nc = x.size()
        xd = nn.functional.dropout(x,p=self.dropout_p,training=True)
        xa = self.att(xd)
        x1 = self.inc(xa)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
                
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        xo = self.outc(torch.cat([x,xd], dim=1)) #torch.cat([x]*32, dim=1)+

        xo = xo.permute(0,1,3,2)
        xo = self.linear1(xo)
        xo = xo.reshape(B,8,Nc)
        output = self.linear2(xo)
        output = output.reshape(B,8)
        return output
    
    def use_checkpointing(self):
        self.att = torch.utils.checkpoint(self.att)
        self.linear = torch.utils.checkpoint(self.linear)
