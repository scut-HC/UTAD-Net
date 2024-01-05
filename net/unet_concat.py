# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=4, repeat_num=6, input_dim=1):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(input_dim, conv_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.001)]

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2,
                          kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, inputs):
        h = self.main(inputs)
        out_src = self.conv1(h)  
        out_cls = self.conv2(h) 
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
    

class MixedFusion_Block(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn=nn.LeakyReLU(0.1, inplace=True)):
        super(MixedFusion_Block, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_dim), act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_dim), act_fn,)

    def forward(self, x1, x2):

        fusion_sum = torch.add(x1, x2)
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(
            x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(
            x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(out_fusion)
        out2 = self.layer2(out1)

        return out2


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(ConvNormRelu, self).__init__()
        norm = nn.BatchNorm2d if norm_type == 'batch' else nn.InstanceNorm2d
        if padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0

        self.unit = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  norm(out_channels),
                                  nn.LeakyReLU(0.01))

    def forward(self, inputs):
        return self.unit(inputs)


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True):
        super(UNetConvBlock, self).__init__()

        self.conv1 = ConvNormRelu(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)
        self.conv2 = ConvNormRelu(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                  norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True):
        super(UNetUpSamplingBlock, self).__init__()
        self.deconv = deconv
        if self.deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
                                    ConvNormRelu(in_channels, out_channels))

    def forward(self, *inputs):
        if len(inputs) == 2:
            return self.forward_concat(inputs[0], inputs[1])
        else:
            return self.forward_standard(inputs[0])

    def forward_concat(self, inputs1, inputs2):
        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):
        return self.up(inputs)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_maps=64, levels=4, norm_type='instance', bias=True,
                 use_last_block=True):
        super(UNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.use_last_block = use_last_block

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps

            conv_block = UNetConvBlock(in_features, out_features, norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            in_features = out_features
        if use_last_block:
            self.center_conv = UNetConvBlock(2 ** (levels - 1) * feature_maps, 2 ** levels * feature_maps)

    def forward(self, inputs):
        encoder_outputs = []
        outputs = inputs
        for i in range(self.levels):
            conv = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            encoder_outputs.append(conv)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(conv)
        if self.use_last_block:
            outputs = self.center_conv(outputs)
        return encoder_outputs, outputs


class UNetDecoder(nn.Module):
    def __init__(self, out_channels=4, feature_maps=64, levels=4, norm_type='instance', bias=True, type='local'):
        super(UNetDecoder, self).__init__()

        self.type = type
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()
        self.tanh = nn.Tanh()
        for i in range(levels):
            upconv = UNetUpSamplingBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                         deconv=False, bias=bias)
            self.features.add_module('upconv%d' % (i + 1), upconv)

            conv_block = UNetConvBlock(2 ** (levels - i) * feature_maps, 2 ** (levels - i - 1) * feature_maps,
                                       norm_type=norm_type, bias=bias)
            self.features.add_module('convblock%d' % (i + 1), conv_block)
        self.score = nn.Conv2d(feature_maps, out_channels, kernel_size=1, bias=bias)

    def forward(self, inputs, encoder_outputs=None):
        decoder_outputs = []
        encoder_outputs.reverse()

        outputs = inputs
        for i in range(self.levels):
            upcon = getattr(self.features, 'upconv%d' % (i + 1))(encoder_outputs[i], outputs)
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(upcon)
            decoder_outputs.append(outputs)
        encoder_outputs.reverse()
        return decoder_outputs, self.tanh(self.score(outputs))


class UNet_middle_concat(nn.Module):
    def __init__(self, used_modality_num=1, used_tumor_type=1, all_modality=4, feature_map=64, levels=4, norm_type='instance',
                 bias=True):
        super(UNet_middle_concat, self).__init__()
        self.img_encoder = UNetEncoder(used_modality_num + all_modality, feature_map, levels, norm_type, bias=bias)
        self.tumor_encoder = UNetEncoder(used_tumor_type * used_modality_num + all_modality, feature_map, levels, norm_type, bias=bias)
        self.img_decoder = UNetDecoder(1, feature_map, levels, norm_type, bias=bias)
        self.tumor_decoder = UNetDecoder(1, feature_map, levels, norm_type, bias=bias)
        self.fusion = MixedFusion_Block(in_dim=feature_map * (2 ** levels), out_dim=feature_map * (2 ** levels))
    def forward(self, img, tumor, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, img.size(2), img.size(3))
        input_img = torch.cat([img, c], dim=1)
        input_tumor = torch.cat([tumor, c], dim=1)
        encoder_outputs_img, final_output_img = self.img_encoder(input_img)
        encoder_outputs_tumor, final_output_tumor = self.tumor_encoder(input_tumor)
        concat_future = self.fusion(final_output_img, final_output_tumor)
        _, outputs_img = self.img_decoder(concat_future, encoder_outputs_img)
        _, outputs_tumor = self.tumor_decoder(concat_future, encoder_outputs_tumor)
        return outputs_img, outputs_tumor
    def forward_teacher(self, img, tumor, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, img.size(2), img.size(3))
        input_img = torch.cat([img, c], dim=1)
        input_tumor = torch.cat([tumor, c], dim=1)
        encoder_outputs_img, final_output_img = self.img_encoder(input_img)
        _, final_output_tumor = self.tumor_encoder(input_tumor)
        concat_future = self.fusion(final_output_img, final_output_tumor)
        _, outputs_img = self.img_decoder(concat_future, encoder_outputs_img)
        return outputs_img, concat_future

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func) 

