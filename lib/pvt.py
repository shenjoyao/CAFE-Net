import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
#from pretrained_pth.hardnet_68 import hardnet
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from lib.PSA import PSA_p

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
  

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        
        self.bn_acti = bn_acti
        
        self.conv = nn.Conv2d(nIn, nOut, kernel_size = kSize,
                              stride=stride, padding=padding,
                              dilation=dilation,groups=groups,bias=bias)
        
        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)
            
    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output  
     
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        
        return output

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        

    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0=x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0*self.sigmoid(out)

class DWConv_Mulit(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_Mulit, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_Mulit(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x,g, H, W):
        B, N, C = x.shape
        q = self.q(g).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, q, k

class Block(nn.Module):

    def __init__(self, dim, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, g, H, W):
        msa, q, k = self.attn(self.norm1(x),self.norm2(g), H, W)
        
        x = x + g + msa
        
        x = x + self.mlp(self.norm2(x), H, W)

        return x, q, k

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FSEM(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.bn1=nn.BatchNorm2d(nIn)

        self.psa1=PSA_p(nIn,nIn)

        self.conv2_1 =conv2d(nIn, nIn//2, (3,3),padding = 1,dilation=(1,1),act=False)
        self.conv2_2 =conv2d(nIn//2, nIn//2, (3,3),padding = 1,dilation=(1,1),act=False)

        self.conv3_1 =conv2d(nIn, nIn //2, (3,3),padding = 1,dilation=(1,1),act=False)
        self.conv3_2 =conv2d(nIn //2, nIn //4, (3,3),padding = 2,dilation=(2,2),act=False)
        self.conv3_3 =conv2d(nIn //4, nIn //4, (3,3),padding = 3,dilation=(3,3),act=False)

        self.conv4_1 =conv2d(nIn, nIn //2, (3,3),padding = 1,dilation=(1,1),act=False)
        self.conv4_2 =conv2d(nIn //2, nIn //4, (3,3),padding = 3,dilation=(3,3),act=False)
        self.conv4_3 =conv2d(nIn //4, nIn //4, (3,3),padding = 5,dilation=(5,5),act=False)

        self.conv_out=conv2d(nIn,nIn)
        self.selayer = SELayer(nIn)
        

    def forward(self, x):
        o1_2 = self.psa1(x)
        
        o2_1 = self.conv2_1(x)
        o2_2 = self.conv2_2(o2_1)
        
        o3_1 = self.conv3_1(x)
        o3_2 = self.conv3_2(o3_1)
        o3_3 = self.conv3_3(o3_2)

        o4_1 = self.conv4_1(x)
        o4_2 = self.conv4_2(o4_1)
        o4_3 = self.conv4_3(o4_2)
        
        o4=torch.cat([o4_1,o4_2,o4_3],1)
        o3_4=torch.cat([o3_1,o3_2,o3_3],1)
        o2_3=torch.cat([o2_1,o2_2],1)

        x_out=self.bn1(o4+o3_4+o2_3)
        x_out=self.selayer(x_out)+o1_2
        x_out=self.conv_out(x_out)

        return x_out

class CADM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CADM, self).__init__()

        self.patch_embed = OverlapPatchEmbed(img_size=224 // 4, patch_size=3, stride=1, in_chans=in_planes,
                                             embed_dim=out_planes)
        self.patch_embed_g = OverlapPatchEmbed(img_size=224 // 4, patch_size=3, stride=2, in_chans=out_planes,
                                             embed_dim=out_planes)
        self.block = Block(dim=out_planes)
        self.norm = nn.LayerNorm(out_planes)
        self.conv_xa=BasicConv2d(in_planes,out_planes,3,padding=1,stride=1)
        self.ChannelAttention=ChannelAttention(out_planes)
        self.upsample_out = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2=conv2d(out_planes,out_planes)

    def forward(self, x,g):
        B = x.shape[0]
        x_a=self.conv_xa(x)
        x_t, H, W = self.patch_embed(x) 
        g_t,H,W=self.patch_embed_g(g)
        x_t, q, k = self.block(x_t,g_t, H, W)
        x_t = self.norm(x_t)
        x_t = x_t.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_c = self.ChannelAttention(x_a)
        x_t=self.conv2(x_t*x_c)
        x_t=self.upsample_out(x_t)*g
 
        return x_t

class multiscale_feature_aggregation(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c2 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c3 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)
        

        self.c12_11=conv2d(out_c*2, out_c)
        self.c12_12=conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c12_21=conv2d(out_c*2, out_c)
        self.c12_22=conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c22 = conv2d(2*out_c, out_c)
        self.c23 = conv2d(out_c, out_c)

    def forward(self, x1, x2, x3):
        x1 = self.up_1(x1)
        x1 = self.c1(x1)

        x2 = self.c2(x2)
        x2=self.up_2(x2)
        x12=torch.cat([x1,x2],1)
        x12=self.up_2_1(x12)
        
        x12_1=self.c12_11(x12)
        x12_1=self.c12_12(x12_1)
        
        x12_2=self.c12_21(x12)
        x12_2=self.c12_22(x12_2)

        x3=self.up_3(x3)
        x3_1 = self.c3(x3)
        x3_1=(x3_1*x12_1)+x12_2
        x=torch.cat([x3_1,x3],1)
        x=self.c23(self.c22(x))
        return x

class CAFE(nn.Module):
    def __init__(self, channel=32):
        super(CAFE, self).__init__()

        self.backbone = pvt_v2_b2()  
        path = './pretrained_pth/pvt_v2_b2.pth' 
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.fsem1=FSEM(64)
        self.fsem2=FSEM(128)
        self.fsem3=FSEM(320)
        
        self.CADM3=CADM(512,320)
        self.CADM2=CADM(320,128)
        self.CADM1=CADM(128,64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.MFA=multiscale_feature_aggregation([320,128,64],64)

        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(128, 1, 1)
        self.out3 = nn.Conv2d(320, 1, 1)
        self.out4 = nn.Conv2d(512, 1, 1)
        self.out5=nn.Conv2d(64, 1, 1)

    def forward(self, x):

        pvt = self.backbone(x)
        x_1 = pvt[0] 
        x_2 = pvt[1] 
        x_3 = pvt[2] 
        x_4 = pvt[3] 
        x_1=self.fsem1(x_1)
        x_2=self.fsem2(x_2)
        x_3=self.fsem3(x_3)
        x_pr_3_1=self.CADM3(x_4,x_3)
        x_pr_2_1=self.CADM2(x_pr_3_1,x_2)
        x_pr_1_1=self.CADM1(x_pr_2_1,x_1)
        x_out=self.MFA(x_pr_3_1,x_pr_2_1,x_pr_1_1)
        x_out = F.interpolate(self.out5(x_out), scale_factor=2, mode='bilinear')
        prediction1_4 = F.interpolate(self.out1(x_pr_1_1), scale_factor=4, mode='bilinear')
        prediction2_8 = F.interpolate(self.out2(x_pr_2_1), scale_factor=8, mode='bilinear')
        prediction3_16 = F.interpolate(self.out3(x_pr_3_1), scale_factor=16, mode='bilinear')
        prediction4_32 = F.interpolate(self.out4(x_4), scale_factor=32, mode='bilinear')

        return x_out,prediction1_4, prediction2_8, prediction3_16, prediction4_32

if __name__ == '__main__':
    model = CAFE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
