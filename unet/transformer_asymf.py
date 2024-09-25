""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class TransformerAsymf(nn.Module):
    def __init__(self, n_channels, n_classes, input_size, in_dropout_p=0.3, dropout_p=0.3, up_kernel_size=2):
        super(TransformerAsymf, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.att = Transformer(n_heads=1, n_embd=1024, n_blocks=16, p_drop=dropout_p, n_token=input_size[1], n_embd_in=input_size[0], n_channel_out=n_classes)

        self.linear = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(input_size[0], 5))
        self.softmax = nn.Softmax(dim=1)
        self.dropout_p = in_dropout_p

    def forward(self, x: torch.Tensor):
        B,C,Nr,Nc = x.size()
        xd = nn.functional.dropout(x,p=self.dropout_p,training=True)
        xa = self.att(xd)
        xa = xa.permute(0,1,3,2)
        xa = self.linear(xa)
        logits = self.softmax(xa)
        logits = logits.reshape(B,5,Nc).permute(0,2,1)
        return logits
    
    def use_checkpointing(self):
        self.att = torch.utils.checkpoint(self.att)
        self.linear = torch.utils.checkpoint(self.linear)
