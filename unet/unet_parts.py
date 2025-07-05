""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    """Change to layernorm"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )
        self.resconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # channel_factor = self.out_channels // self.in_channels + 1
        # x = self.conv2(self.conv1(x)) + torch.cat([x]*channel_factor,dim=1)[:,:self.out_channels,:,:]
        x = self.norm(x)
        x = self.conv(x) + self.resconv(x)
        return x

def pos_enc(idx : torch.Tensor , n_embd):
    assert n_embd % 2 == 0
    omega = (10000 ** (-2 * torch.arange(n_embd // 2) / n_embd)).view(1,-1)
    pe = torch.cat([torch.sin(idx.view(-1,1)*omega),torch.cos(idx.view(-1,1)*omega)], dim = -1).view(-1,n_embd)
    return pe

class Attention(nn.Module): 
    """Downscaling with maxpool then doing attention"""

    def __init__(self, in_channel, p_drop = 0.2, n_embd = 1024, n_heads=2, patch_size=4, N=32):
        super().__init__()

        # calculate all dimensions
        self.N = N # the input image is NxN (Batch,Channel,N,N)
        M = N//patch_size #image is devided into MxM patches
        self.n_patch = in_channel * patch_size **2 # flatten patch size
        self.n_embd = n_embd # attention embedding dimension

        # initiate modules for patch and unpatch
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = torch.nn.Fold(N,kernel_size=patch_size, stride=patch_size) # fold the sequency back to 32x32 images

        # intitiate the attention modules
        self.c_attn = nn.Linear(self.n_patch, 3 * self.n_embd, bias=False) # linear block for finding QKV vectors
        self.mlp = nn.Sequential(nn.Linear(self.n_embd, self.n_patch, bias=False),
                                nn.ReLU(inplace=True)) # up coversion 
        self.mhatt = nn.MultiheadAttention(embed_dim = self.n_embd, num_heads = n_heads, dropout= p_drop, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.ln = nn.GroupNorm(1,M*M) #layernorm for attention block, M*M is the sequence length

        # positional encoding        
        x_enc = torch.arange(M).repeat(M)
        y_enc = torch.arange(M).view(-1,1).repeat([1,M]).flatten()
        x_enc = pos_enc(x_enc, self.n_patch//2)
        y_enc = pos_enc(y_enc, self.n_patch//2)
        self.register_buffer('pe', torch.cat([x_enc, y_enc], dim=-1)[None,:,:], persistent=False)  #2D position encoding, first half is for x-pos, second half is for y-pos

    def forward(self, x):
        # assetion of the input size
        B, C, H, W = x.size()
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        assert H == self.N and W == self.N

        # message the input image patchify the image for attention
        xp = self.unfold(x) # x: B, C*p*p, L
        xp = xp.permute(0,2,1).contiguous() # x: B,L,C*p*p = B,T,E (batch size, sequence length, embedding dimensionality:n_embd)
        xp += self.pe #adding position encoding
        xp = self.ln(xp)
        # attention block
        q, k, v  = self.c_attn(xp).split(self.n_embd, dim=2)
        y, _ = self.mhatt(q, k, v)
        y = self.relu(y)
        y = self.mlp(y)
        y = self.fold(y.permute(0,2,1)) # B,L,C*p*p -> x: B,C*p*p,L
        
        x += y
        return x

class Transformer(nn.Module):
    """adding positional encoding and multiple Transformer blocks"""

    def __init__(self, n_heads=1, n_embd = 1024, n_blocks=5, p_drop = 0.2, n_token=408, n_embd_in=512, n_channel_out =4):
        super().__init__()

        # calculate all dimensions
        self.n_token = n_token # number of tokens: number of row or number of col
        self.n_embd = n_embd # attention embedding dimension
        self.n_embd_in = n_embd_in
        self.n_channel_out = n_channel_out

        #Linear projection
        self.in_linear = nn.Linear(self.n_embd_in, self.n_embd)
        self.out_linear = nn.Linear(self.n_embd, self.n_embd_in * n_channel_out)

        #Transformer train
        blocks = [Transformer_element(n_heads=n_heads, n_embd=n_embd, n_token=n_token, p_drop=p_drop) for _ in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # positional encoding
        p_e =pos_enc(torch.arange(self.n_token), self.n_embd) 
        self.register_buffer('pe',p_e[None,:,:], persistent=False)  #2D position encoding, first half is for x-pos, second half is for y-pos

    def forward(self, x: torch.Tensor):
        # assetion of the input size
        B, C, E, T = x.size()
        assert C == 1
        assert T == self.n_token
        assert E == self.n_embd_in

        # Transpose the array, each col is a token. n_col is n_token
        xp = x.reshape(B,E,T).permute(0,2,1)

        # Linear projection, change the embedding dimension
        xp = self.in_linear(xp)
        xp = xp + self.pe #adding position encoding
        # transformer blocks
        xp = self.blocks(xp)
        # Linear projection, outputs n_channel_out channels, so n_channel_out * n_embd_in
        xp = self.out_linear(xp)
        # First reshape to channel last, then permute the axes.
        xout = xp.reshape(B, T, E, self.n_channel_out).permute(0,3,2,1)
        xout = xout + torch.cat([x]*self.n_channel_out, dim=1) #resdual connection
        return xout
    
class Transformer_element(nn.Module):
    """attention + mlp"""

    def __init__(self, n_heads=1, n_embd = 1024, n_token=512, p_drop = 0.2):
        super().__init__()

        # calculate all dimensions
        self.n_token = n_token # number of tokens number of row or number of col
        self.n_embd = n_embd # attention embedding dimension

        # intitiate the attention modules
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False) # linear block for finding QKV vectors
        self.mlp = nn.Sequential(nn.Linear(self.n_embd, self.n_embd, bias=False),
                                nn.ReLU(inplace=True)) 
        self.mhatt = nn.MultiheadAttention(embed_dim = self.n_embd, num_heads = n_heads, dropout= p_drop, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.ln = nn.GroupNorm(1,n_token) #layernorm for attention block, M*M is the sequence length

    def forward(self, x):
        xp = self.ln(x) #layernorm
        # attention block
        q, k, v  = self.c_attn(xp).split(self.n_embd, dim=2)
        y, _ = self.mhatt(q, k, v)
        y = self.relu(y)
        y = self.mlp(y)
        
        y = x + y# residual connection
        return y
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.GroupNorm(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.ln = nn.GroupNorm(in_channels, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.ln(x1)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        y = self.conv1(x) + x
        y = self.conv2(y)
        return y
