import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor):
        return F.gelu(input)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        nn.init.constant_(self.proj.weight.data, 0.0)
        nn.init.constant_(self.proj.bias.data, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        B_q, N_q, _ = q.size()
        B_k, N_k, _ = k.size()
        q = self.q(q).reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.attn_drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        q = (attn @ v).transpose(1, 2).reshape(q.size(0), q.size(2), -1)
        q = self.proj_drop(self.proj(q))
        return q


class Cross(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.self_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=2 * dim, out_features=dim, drop=drop)

    def forward(self, q, x):
        q_bn = self.bn1(q)
        q = q + self.drop_path(self.cross_attn(q_bn, x, x))
        q = q + self.drop_path(self.mlp(q))
        q_bn = self.bn2(q)
        q = q + self.drop_path(self.self_attn(q_bn, q_bn, q_bn))
        return q


class Cross_Refine(nn.Module):
    def __init__(self, num_heads, mid_dim=1024, encoder_layer=1, cross_layer=2, qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()

        self.mid_dim = mid_dim
        self.token_norm = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.LayerNorm(mid_dim))
        self.cross = nn.ModuleList([Cross(mid_dim, num_heads, qkv_bias, drop, attn_drop, drop_path) for _ in range(cross_layer)])
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=mid_dim, kernel_size=(1, 1), stride=1, padding=0), nn.BatchNorm2d(mid_dim))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=mid_dim, out_channels=2048, kernel_size=(3, 3), stride=2, padding=1), nn.BatchNorm2d(2048))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, local_x: Tensor, global_x: Tensor):
        '''
            x: local feature map    [B, 1024, 14, 14]
            global_x : by Resnet50  [B, 2048, 7, 7]
        '''
        B, _, H, W = local_x.size()
        B2,_,H2,W2 = global_x.size()

        local_x = local_x.reshape(B, self.mid_dim, H * W).permute(0, 2, 1) # torch.Size([16, 196, 1024])
        global_x = self.conv1(global_x).reshape(B2, self.mid_dim, H2 * W2).permute(0, 2, 1) # torch.Size([16, 49, 1024])
        
        # token = local_x channel = q
        token = local_x
        
        for cross in self.cross: 
            token = cross(token, global_x) # torch.Size([16, 196, 1024])
        
        token = self.conv2(token.permute(0, 2, 1).view(B, -1, H, W))

        pool_x = self.avgpool(token)
        pool_x = torch.flatten(pool_x, 1)
        return pool_x # [B, 2048, 7, 7]



if __name__ == '__main__':
    model = Cross_Refine(8)
    local_x = torch.Tensor(16, 1024, 14, 14)
    global_x = torch.Tensor(16, 2048, 7, 7)

    outputs = model(local_x, global_x)
    print(outputs.shape)

