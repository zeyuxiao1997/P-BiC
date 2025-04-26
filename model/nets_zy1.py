import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
import torch.nn.init as init
import functools
from einops import rearrange
import numbers


class my_model(nn.Module):
    def __init__(self,
                 en_feature_num,
                 en_resblock_num,
                 num_patch,
                 ):
        super(my_model, self).__init__()
        self.en_feature_num = en_feature_num
        self.en_resblock_num = en_resblock_num
        self.num_patch = num_patch

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        ResidualBlock_L1 = functools.partial(ResidualBlock_noBN, nf=self.en_feature_num)
        ResidualBlock_L2 = functools.partial(ResidualBlock_noBN, nf=2*self.en_feature_num)
        ResidualBlock_L3 = functools.partial(ResidualBlock_noBN, nf=3*self.en_feature_num)

        self.conv_first_RGB = nn.Conv2d(12, self.en_feature_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_first_green = nn.Conv2d(4, self.en_feature_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.extraction_L1_RGB = make_layer(ResidualBlock_L1, self.en_resblock_num)
        self.extraction_L1_green = make_layer(ResidualBlock_L1, self.en_resblock_num)
        self.fea_L2_RGB = nn.Conv2d(self.en_feature_num, 2*self.en_feature_num, 3, 2, 1, bias=True)
        self.fea_L2_green = nn.Conv2d(self.en_feature_num, 2*self.en_feature_num, 3, 2, 1, bias=True)
        self.extraction_L2_RGB = make_layer(ResidualBlock_L2, self.en_resblock_num)
        self.extraction_L2_green = make_layer(ResidualBlock_L2, self.en_resblock_num)
        self.fea_L3_RGB = nn.Conv2d(2*self.en_feature_num, 3*self.en_feature_num, 3, 2, 1, bias=True)
        self.fea_L3_green = nn.Conv2d(2*self.en_feature_num, 3*self.en_feature_num, 3, 2, 1, bias=True)
        self.extraction_L3_RGB = make_layer(ResidualBlock_L3, self.en_resblock_num)
        self.extraction_L3_green = make_layer(ResidualBlock_L3, self.en_resblock_num)

        ###############################################
        self.DualTransfer = DualTransfer(nf=3*self.en_feature_num, stride=self.num_patch, resblock_num=2)
        self.fuse = nn.Conv2d(9*self.en_feature_num, 3*self.en_feature_num, 3, 1, 1, bias=True)
        self.fuse_resblock = make_layer(ResidualBlock_L3, self.en_resblock_num)
        ###############################################

        self.fuse_L2 = nn.Conv2d(4*self.en_feature_num, 2*self.en_feature_num, 3, 1, 1, bias=True)
        self.fuse_L1 = nn.Conv2d(2*self.en_feature_num, self.en_feature_num, 3, 1, 1, bias=True)
        
        self.upconv3_i = nn.Conv2d(3*self.en_feature_num, 3*self.en_feature_num, 3, 1, 1, bias=True)
        # self.upconv3_3 = resnet_block(3*self.en_feature_num)
        self.upconv3_2 = resnet_block(3*self.en_feature_num)
        self.upconv3_1 = resnet_block(3*self.en_feature_num)

        self.upconv2_u = upconv(3*self.en_feature_num, 2*self.en_feature_num)
        self.upconv2_i = nn.Conv2d(4*self.en_feature_num, 2*self.en_feature_num, 3, 1, 1, bias=True)
        # self.upconv2_3 = resnet_block(2*self.en_feature_num)
        self.upconv2_2 = resnet_block(2*self.en_feature_num)
        self.upconv2_1 = resnet_block(2*self.en_feature_num)

        self.upconv1_u = upconv(2*self.en_feature_num, self.en_feature_num)
        self.upconv1_i = nn.Conv2d(2*self.en_feature_num, self.en_feature_num, 3, 1, 1, bias=True)
        # self.upconv1_3 = resnet_block(self.en_feature_num)
        self.upconv1_2 = resnet_block(self.en_feature_num)
        self.upconv1_1 = resnet_block(self.en_feature_num)

        self.img_prd_L3 = nn.Conv2d(3*self.en_feature_num, 12, 3, 1, 1, bias=True)
        self.img_prd_L2 = nn.Conv2d(2*self.en_feature_num, 12, 3, 1, 1, bias=True)
        self.img_prd_L1 = nn.Conv2d(self.en_feature_num, 12, 3, 1, 1, bias=True)


    def forward(self, RGB, green):
        H, W = RGB.shape[2:]
        RGB = self.check_image_size(RGB)
        green = self.check_image_size(green)

        RGBpx = F.pixel_unshuffle(RGB, 2)
        greenpx = F.pixel_unshuffle(green, 2)

        RGB_L2 = F.interpolate(RGBpx, scale_factor=1/2, mode='bilinear')
        green_L2 = F.interpolate(greenpx, scale_factor=1/2, mode='bilinear')
        RGB_L3 = F.interpolate(RGB_L2, scale_factor=1/2, mode='bilinear')
        green_L3 = F.interpolate(green_L2, scale_factor=1/2, mode='bilinear')

        ########################################################################
        # L1
        L1_fea_RGB = self.lrelu(self.conv_first_RGB(RGBpx))
        L1_fea_RGB = self.extraction_L1_RGB(L1_fea_RGB)
        L1_fea_green = self.lrelu(self.conv_first_green(greenpx))
        L1_fea_green = self.extraction_L1_green(L1_fea_green)
        # L2
        L2_fea_RGB = self.lrelu(self.fea_L2_RGB(L1_fea_RGB))
        L2_fea_RGB = self.extraction_L2_RGB(L2_fea_RGB)
        L2_fea_green = self.lrelu(self.fea_L2_green(L1_fea_green))
        L2_fea_green = self.extraction_L2_green(L2_fea_green)
        # L3
        L3_fea_RGB = self.lrelu(self.fea_L3_RGB(L2_fea_RGB))
        L3_fea_RGB = self.extraction_L3_RGB(L3_fea_RGB)
        L3_fea_green = self.lrelu(self.fea_L3_green(L2_fea_green))
        L3_fea_green = self.extraction_L3_green(L3_fea_green)
        #############################################################
        green_px_resize = F.interpolate(green_L3, scale_factor=1/self.num_patch, mode='bilinear')
        # RGB_px_resize = F.interpolate(RGB_L3, scale_factor=1/self.num_patch, mode='bilinear')
        green_fea_resize = F.interpolate(L3_fea_green, scale_factor=1/self.num_patch, mode='bilinear')
        # RGB_fea_resize = F.interpolate(L3_fea_RGB, scale_factor=1/self.num_patch, mode='bilinear')
        green_fea_patchs = crop(L3_fea_green, self.num_patch)
        RGB_fea_patchs = crop(L3_fea_RGB, self.num_patch)
        green_px_patchs = crop(green_L3, self.num_patch)
        RGB_px_patchs = crop(RGB_L3, self.num_patch)

        out = self.DualTransfer(green_px_resize, green_fea_resize,green_fea_patchs,\
                          green_px_patchs, RGB_fea_patchs, RGB_px_patchs)
        fuse = self.lrelu(self.fuse(torch.cat((out,L3_fea_RGB, L3_fea_green),dim=1)))
        fuse = self.fuse_resblock(fuse)

        L2_de = self.fuse_L2(torch.cat((L2_fea_RGB, L2_fea_green),dim=1))
        L1_de = self.fuse_L1(torch.cat((L1_fea_RGB, L1_fea_green),dim=1))
        
        cat3 = self.upconv3_i(fuse)
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(cat3)))
        cat2 = self.upconv2_i(cat_with_crop(L2_de, [L2_de, upconv2]))
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(cat2)))
        cat1 = self.upconv1_i(cat_with_crop(L1_de, [L1_de, upconv1]))

        out_1 = self.img_prd_L1(cat1)
        out_2 = self.img_prd_L2(cat2)
        out_3 = self.img_prd_L3(cat3)

        out_1 = F.pixel_shuffle(out_1, 2)
        out_2 = F.pixel_shuffle(out_2, 2)
        out_3 = F.pixel_shuffle(out_3, 2)

        return out_1[:,:,:H,:W], out_2[:,:,:H,:W], out_3[:,:,:H,:W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (32 - h % 32) % 32
        mod_pad_w = (32 - w % 32) % 32
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output

def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def resnet_block(in_channels, kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


class LocalGlobalFusion(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias'):
        super(LocalGlobalFusion, self).__init__()

        self.norm1_RGB = LayerNorm(dim, LayerNorm_type)
        self.norm1_green = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, RGB, green):
        assert RGB.shape == green.shape, 'the shape doesnt equal' 
        b, c , h, w = RGB.shape
        fused = RGB + self.attn(self.norm1_RGB(RGB), self.norm1_green(green)) # b, c, h, w
        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused


class Green2RGB(nn.Module):
    def __init__(self, nf, stride, num_heads):
        super(Green2RGB, self).__init__()
        self.num_heads = num_heads
        self.conv_fgreen = nn.Conv2d(nf+1*4, nf, 3, 1, 1, bias=True)
        self.conv_fRGB = nn.Conv2d(nf+3*4, nf, 3, 1, 1, bias=True)
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_a = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_b = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_cat1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_green = nn.Conv2d(nf+1*4, nf, 3, 1, 1, bias=True)
        
        self.LocalGlobalFusion = LocalGlobalFusion(nf, num_heads=self.num_heads,
                                                   ffn_expansion_factor=4, bias=False,
                                                   LayerNorm_type='WithBias')

    def forward(self, RGB, green, RGBpix, greenpix, greenFeaResize, greenpixResize):
        # print(RGB.shape, green.shape, RGBpix.shape, greenpix.shape, greenFeaResize.shape, greenpixResize.shape)
        green = self.conv_fgreen(torch.cat((green, greenpix), dim=1))
        RGB = self.conv_fRGB(torch.cat((RGB, RGBpix), dim=1))
        
        res_a = self.lrelu(self.conv_down_a(green)) - RGB
        out_a = self.lrelu(self.conv_up_a(res_a)) + green

        res_b = RGB - self.lrelu(self.conv_down_b(green))
        out_b = self.lrelu(self.conv_up_b(res_b + RGB)) 
        out = self.lrelu(self.conv_cat1(torch.cat([out_a, out_b], dim=1)))
        out = self.LocalGlobalFusion(out, self.conv_green(torch.cat((greenFeaResize,greenpixResize),dim=1)))
        return out
    
class RGB2Green(nn.Module):
    def __init__(self, nf, resblock_num=2):
        super(RGB2Green, self).__init__()
        self.conv_fgreen = nn.Conv2d(nf+1*4, nf, 3, 1, 1, bias=True)
        self.conv_fRGB = nn.Conv2d(nf+3*4, nf, 3, 1, 1, bias=True)
        self.conv_fuse = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        ResidualBlock = functools.partial(ResidualBlock_noBN, nf=nf)
        self.resblock = make_layer(ResidualBlock, resblock_num)
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        self.conv_shared = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv_refine = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # initialization
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()

    def forward(self, RGB, green, RGBpix, greenpix):
        green = self.conv_fgreen(torch.cat((green, greenpix), dim=1))
        RGB = self.conv_fRGB(torch.cat((RGB, RGBpix), dim=1))

        fuse = self.conv_fuse(torch.cat([RGB, green], dim=1))
        fuse = self.resblock(fuse) - RGB
        RGB = self.conv(fuse)
        ###
        green_normed = self.norm_layer(green)
        style = self.lrelu(self.conv_shared(torch.cat([RGB, green], dim=1)))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)
        b, c, h, w = RGB.size()
        RGB = RGB.view(b, c, h * w)
        RGB_mean = torch.mean(RGB, dim=-1, keepdim=True).unsqueeze(3)
        RGB_std = torch.std(RGB, dim=-1, keepdim=True).unsqueeze(3)
        ###
        gamma = gamma + RGB_std
        beta = beta + RGB_mean
        out = green_normed * gamma + beta

        out = self.conv_refine(out) + out

        return out


class DualTransfer(nn.Module):
    def __init__(self, nf, stride, resblock_num):
        super(DualTransfer, self).__init__()
        self.stride = stride
        self.Green2RGB = Green2RGB(nf=nf, stride=self.stride, num_heads=1)
        self.RGB2Green = RGB2Green(nf, resblock_num)
        self.fuse = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)

    def forward(self, green_px_resize, green_fea_resize,\
                      green_fea_patchs, green_px_patchs,\
                      RGB_fea_patchs, RGB_px_patchs):
        out = []
        for i in range(len(green_fea_patchs)):
            out1 = self.Green2RGB(RGB_fea_patchs[i],green_fea_patchs[i],
                                  RGB_px_patchs[i], green_px_patchs[i],
                                  green_fea_resize, green_px_resize)
            out2 = self.RGB2Green(RGB_fea_patchs[i],green_fea_patchs[i],
                                  RGB_px_patchs[i], green_px_patchs[i])
            tmp = self.fuse(torch.cat((out1, out2), dim=1))
            out.append(tmp)
        
        out = merge(out, self.stride)
        return out



def crop(x, num):
    b,c,h,w = x.shape
    crops = []
    h_c = int(h/num)
    w_c = int(w/num)
    for i in range(num):
        for j in range(num):
            tmp = x[:,:,i*h_c:i*h_c+h_c,j*w_c:j*w_c+w_c]
            crops.append(tmp.cuda())
    return crops

def merge(x, num):
    b,c,h,w = x[0].shape
    merge = torch.zeros(b,c, h*num, w*num)
    i = 0
    j = 0
    iter = 0
    for item in x:
        i = iter//num
        j = iter-i*num
        merge[:,:,i*h:i*h+h,j*w:j*w+w] = item
        iter+=1
 
    return merge.cuda()


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1, y_2, y_3):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3

 
class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

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

class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps are not equal!'
        b,c,h,w = x.shape
        q = self.q(x) # image
        k = self.k(y) # full
        v = self.v(y) # full
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

