import torch
import torch.nn as nn
import math
import itertools

from timm.models.layers import DropPath


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
    
        # global FLOPS_COUNTER
        # output_points = ((resolution + 2 * padding - dilation *
        #                   (ks - 1) - 1) // stride + 1)**2
        # FLOPS_COUNTER += a * b * output_points * (ks**2) // groups
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class InvertResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.pwconv1_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=1,  stride=1, padding=0)
        self.dwconv_bn = Conv2d_BN(self.hidden_features, self.hidden_features, kernel_size=3,  stride=1, padding=1, groups= self.hidden_features)
        self.pwconv2_bn = Conv2d_BN(self.hidden_features, self.in_features, kernel_size=1,  stride=1, padding=0)

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # @line_profile
    def forward(self, x):
        x1 = self.pwconv1_bn(x)
        x1 = self.act(x1)
        x1 = self.dwconv_bn(x1)
        x1 = self.act(x1)
        x1 = self.pwconv2_bn(x1)

        return x + x1

class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=2, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5
        
        self.nh_kd = key_dim * num_heads
        self.qk_dim = 2 * self.nh_kd
        self.v_dim = int(attn_ratio * key_dim) * num_heads
        dim_h = self.v_dim + self.qk_dim

        self.N = resolution ** 2
        self.N2 = self.N
        self.pwconv = nn.Conv2d(dim, dim_h, kernel_size=1,  stride=1, padding=0)
        self.dwconv = Conv2d_BN(self.v_dim, self.v_dim, kernel_size=3,  stride=1, padding=1, groups=self.v_dim)
        self.proj_out = nn.Linear(self.v_dim, dim)
        self.act = nn.GELU()

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        h, w = self.resolution, self.resolution
        x = x.transpose(1, 2).reshape(B, C, h, w)

        x = self.pwconv(x)
        qk, v1 = x.split([self.qk_dim, self.v_dim], dim=1)
        qk = qk.reshape(B, 2, self.num_heads, self.key_dim, N).permute(1, 0, 2, 4, 3)
        q, k = qk[0], qk[1]

        v1 = v1 + self.act(self.dwconv(v1))
        v = v1.reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.v_dim)
        x = self.proj_out(x)
        return x

class ModifiedTransformer(nn.Module):  
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio= 2, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = Attention(dim=self.dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, resolution=resolution)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = Mlp(in_features=self.dim, hidden_features=self.dim*mlp_ratio, out_features=self.dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SBCFormerBlock(nn.Module):   # building block
    def __init__(self, dim, depth_invres=2, depth_mattn=4, depth_mixer=2, key_dim=24, num_heads=6, mlp_ratio=2, attn_ratio=2, drop=0., attn_drop=0.,
                 drop_paths=[0.02500000037252903, 0.03750000149011612, 0.05000000074505806, 0.0625], act_layer=nn.GELU, pool_ratio=2, invres_ratio=2, resolution=7):
        super().__init__()

        # print(depth_invres, depth_mattn, depth_mixer, dim) # 2 2 2 128
        # print(key_dim, num_heads, mlp_ratio, attn_ratio) # 24 4 4 2
        # print(drop, attn_drop,drop_paths, act_layer) # 0.0 0.0 [0.0, 0.012500000186264515] <class 'torch.nn.modules.activation.GELU'>
        # print(pool_ratio, invres_ratio, resolution) # 4 2 7

        '''
stages =  0
2 2 2 128
24 4 4 2
0.0 0.0 [0.0, 0.012500000186264515] <class 'torch.nn.modules.activation.GELU'>
4 2 7
stages =  1
2 4 2 256
24 6 4 2
0.0 0.0 [0.02500000037252903, 0.03750000149011612, 0.05000000074505806, 0.0625] <class 'torch.nn.modules.activation.GELU'>
2 2 7
stages =  2
1 3 2 384
24 8 4 2
0.0 0.0 [0.07500000298023224, 0.08749999850988388, 0.10000000149011612] <class 'torch.nn.modules.activation.GELU'>
1 2 7
'''

        self.resolution = resolution
        self.dim = dim
        self.depth_invres = depth_invres
        self.depth_mattn = depth_mattn
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        self.invres_blocks = nn.Sequential()
        for k in range(self.depth_invres):
            self.invres_blocks.add_module("InvRes_{0}".format(k), InvertResidualBlock(in_features=dim, hidden_features=int(dim*invres_ratio), out_features=dim, kernel_size=3, drop_path=0.))

        self.pool_ratio= pool_ratio
        if self.pool_ratio > 1:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            self.convTrans= nn.ConvTranspose2d(dim, dim, kernel_size=pool_ratio, stride=pool_ratio, groups=dim)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.pool = nn.Identity()
            self.convTrans = nn.Identity()
            self.norm = nn.Identity()
        
        self.mixer = nn.Sequential()
        for k in range(self.depth_mixer):
            self.mixer.add_module("Mixer_{0}".format(k), InvertResidualBlock(in_features=dim, hidden_features=dim*2, out_features=dim, kernel_size=3, drop_path=0.))
        
        self.trans_blocks = nn.Sequential()
        for k in range(self.depth_mattn):
            self.trans_blocks.add_module("MAttn_{0}".format(k), ModifiedTransformer(dim=dim, key_dim=key_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
             drop=drop, attn_drop=attn_drop, drop_path=drop_paths[k], resolution=resolution))
        
        self.proj = Conv2d_BN(self.dim, self.dim, kernel_size=1,  stride=1, padding=0)
        self.proj_fuse = Conv2d_BN(self.dim*2, self.dim, kernel_size=1,  stride=1, padding=0)
        
    def forward(self, global_x, local_x):
        '''
        x.shape =  torch.Size([2, 128, 28, 28])
        out.shape =  torch.Size([2, 128, 28, 28])
        x.shape =  torch.Size([2, 256, 14, 14])
        out.shape =  torch.Size([2, 256, 14, 14])
        x.shape =  torch.Size([2, 384, 7, 7])
        out.shape =  torch.Size([2, 384, 7, 7])
        '''
        # print('global_x.shape = ', global_x.shape) #  torch.Size([2, 128, 28, 28])
        # print('local_x.shape = ', local_x.shape)
        B, C, _, _ = global_x.shape
        h, w = self.resolution, self.resolution
        local_fea = local_x
        x = self.invres_blocks(global_x)

        if self.pool_ratio > 1.:
            x = self.pool(x)
        
        x = self.mixer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.trans_blocks(x)
        x = x.transpose(1, 2).reshape(B, C, h, w)

        if self.pool_ratio > 1:
            x = self.convTrans(x)
            x = self.norm(x)
        global_act = self.act(self.proj(x))
        x_ = local_fea * global_act
        x_cat = torch.cat((x, x_), dim=1)
        out = self.proj_fuse(x_cat)

        # print('out.shape = ', out.shape) # torch.Size([2, 128, 28, 28])

        return out


class SBCFormerBlock_test(nn.Module):   # building block
    def __init__(self, dim, depth_invres=2, depth_mattn=4, depth_mixer=2, key_dim=24, num_heads=6, mlp_ratio=2, attn_ratio=2, drop=0., attn_drop=0.,
                 drop_paths=[0.02500000037252903, 0.03750000149011612, 0.05000000074505806, 0.0625], act_layer=nn.GELU, pool_ratio=2, invres_ratio=2, resolution=7):
        super().__init__()

        # print(depth_invres, depth_mattn, depth_mixer, dim) # 2 2 2 128
        # print(key_dim, num_heads, mlp_ratio, attn_ratio) # 24 4 4 2
        # print(drop, attn_drop,drop_paths, act_layer) # 0.0 0.0 [0.0, 0.012500000186264515] <class 'torch.nn.modules.activation.GELU'>
        # print(pool_ratio, invres_ratio, resolution) # 4 2 7

        '''
stages =  0
2 2 2 128
24 4 4 2
0.0 0.0 [0.0, 0.012500000186264515] <class 'torch.nn.modules.activation.GELU'>
4 2 7
stages =  1
2 4 2 256
24 6 4 2
0.0 0.0 [0.02500000037252903, 0.03750000149011612, 0.05000000074505806, 0.0625] <class 'torch.nn.modules.activation.GELU'>
2 2 7
stages =  2
1 3 2 384
24 8 4 2
0.0 0.0 [0.07500000298023224, 0.08749999850988388, 0.10000000149011612] <class 'torch.nn.modules.activation.GELU'>
1 2 7
'''

        self.resolution = resolution
        self.dim = dim
        self.depth_invres = depth_invres
        self.depth_mattn = depth_mattn
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        self.invres_blocks = nn.Sequential()
        for k in range(self.depth_invres):
            self.invres_blocks.add_module("InvRes_{0}".format(k), InvertResidualBlock(in_features=dim, hidden_features=int(dim*invres_ratio), out_features=dim, kernel_size=3, drop_path=0.))

        self.pool_ratio= pool_ratio
        if self.pool_ratio > 1:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            self.convTrans= nn.ConvTranspose2d(dim, dim, kernel_size=pool_ratio, stride=pool_ratio, groups=dim)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.pool = nn.Identity()
            self.convTrans = nn.Identity()
            self.norm = nn.Identity()
        
        self.mixer = nn.Sequential()
        for k in range(self.depth_mixer):
            self.mixer.add_module("Mixer_{0}".format(k), InvertResidualBlock(in_features=dim, hidden_features=dim*2, out_features=dim, kernel_size=3, drop_path=0.))
        
        self.trans_blocks = nn.Sequential()
        for k in range(self.depth_mattn):
            self.trans_blocks.add_module("MAttn_{0}".format(k), ModifiedTransformer(dim=dim, key_dim=key_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
             drop=drop, attn_drop=attn_drop, drop_path=drop_paths[k], resolution=resolution))
        
        self.proj = Conv2d_BN(self.dim, self.dim, kernel_size=1,  stride=1, padding=0)
        self.proj_fuse = Conv2d_BN(self.dim*2, self.dim, kernel_size=1,  stride=1, padding=0)
        
    def forward(self, x):
        '''
        x.shape =  torch.Size([2, 128, 28, 28])
        out.shape =  torch.Size([2, 128, 28, 28])
        x.shape =  torch.Size([2, 256, 14, 14])
        out.shape =  torch.Size([2, 256, 14, 14])
        x.shape =  torch.Size([2, 384, 7, 7])
        out.shape =  torch.Size([2, 384, 7, 7])
        '''
        x  = torch.split(x, (512, 512), dim=2)
        global_x = x[0][0]
        local_x = x[0][1]


        print('global_x.shape = ', global_x.shape) #  torch.Size([2, 128, 28, 28])
        print('local_x.shape = ', local_x.shape)
        B, C, _, _ = global_x.shape
        h, w = self.resolution, self.resolution
        local_fea = local_x
        x = self.invres_blocks(global_x)

        if self.pool_ratio > 1.:
            x = self.pool(x)
        
        x = self.mixer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.trans_blocks(x)
        x = x.transpose(1, 2).reshape(B, C, h, w)

        if self.pool_ratio > 1:
            x = self.convTrans(x)
            x = self.norm(x)
        global_act = self.act(self.proj(x))
        x_ = local_fea * global_act
        x_cat = torch.cat((x, x_), dim=1)
        out = self.proj_fuse(x_cat)

        print('out.shape = ', out.shape) # torch.Size([2, 128, 28, 28])

        return out

if __name__ == '__main__':
    from torchsummary import summary

    model = SBCFormerBlock_test(dim=512)
    x = torch.Tensor(16, 2048, 14, 14)
    summary(model, (16, 1024, 14, 14))
    out = model(x)