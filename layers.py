import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
#from customFCGoogleSlow import CustomFullyConnectedLayer as customLinear
from customFCGoogleSlowParallel import CustomFullyConnectedLayer as customLinear

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.,sparsity:float=0.1):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout,sparsity=sparsity)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            customLinear(feats, mlp_hidden,sparsity=sparsity),
            nn.GELU(),
            nn.Dropout(dropout),
            customLinear(mlp_hidden, feats,sparsity=sparsity),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.,sparsity:float=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = customLinear(feats, feats,sparsity=sparsity)
        self.k = customLinear(feats, feats,sparsity=sparsity)
        self.v = customLinear(feats, feats,sparsity=sparsity)

        self.o = customLinear(feats, feats,sparsity=sparsity)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)



