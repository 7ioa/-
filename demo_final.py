from __future__ import division
import os,time,cv2

import numpy as np
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
# 用于记录日志
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt

max_epoch_num = 20
max_test_num = 12000  
mini_batch_size = 30
NO_USE_NORMALIZATION = 0     
# is_training = True
is_training = True
max_patch_num = 140000
trainImageSize = 128
ReadColorImage=1 
isJointTrain = False
lambda1 = 100
lambda2 = 10

##################################################################################
def create_logger(log_file):
    # 定义好logger
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler() # 日志输出到流
    console.setLevel(logging.INFO) # 日志等级
    console.setFormatter(logging.Formatter(log_format)) # 设置日志格式
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return F1

def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}

# 定义dataset
class G1G2Dataset(torch_data.Dataset):
    def __init__(self, mode, target_size=(128,128)):
        self.mode = mode
        self.target_size = target_size
        if self.mode == 'train':
            self.imageset_dir = os.path.join('./data/training/image')
            self.imageset_gt_dir = os.path.join('./data/training/mask')
        elif self.mode == 'test_MDvsFA':
            self.imageset_dir = os.path.join('./data/test_MDvsFA/image')
            self.imageset_gt_dir = os.path.join('./data/test_MDvsFA/mask')
        elif self.mode == 'test_SIRST':
            self.imageset_dir = os.path.join('./data/test_SIRST/image')
            self.imageset_gt_dir = os.path.join('./data/test_SIRST/mask')
        else:
            raise NotImplementedError
        # 读取文件夹中的所有文件名
        self.image_filenames = sorted(os.listdir(self.imageset_dir))
        self.gt_filenames = sorted(os.listdir(self.imageset_gt_dir))

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_filenames)
        elif self.mode == 'test_MDvsFA':
            return len(self.gt_filenames) 
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_dir = os.path.join(self.imageset_dir, self.image_filenames[idx])
            gt_dir = os.path.join(self.imageset_gt_dir, self.gt_filenames[idx])
            
            # -1表示同时加载了alpha通道
            real_input = np.float32(cv2.imread(img_dir,-1))/255.0

            if  ReadColorImage == 0:
                input_images = real_input * 2 - 1
            else:
                input_images = real_input[:,:,2] * 2 - 1
                input_images = np.expand_dims(input_images,axis=0)
            
            bufImg = cv2.imread(gt_dir, -1)
            dilated_bufImg = bufImg
            output_images = np.float32(dilated_bufImg)/255.0 # 像素归一化
            output_images = np.expand_dims(output_images,axis=0)
            

            sample_info = {}
            sample_info['input_images'] = input_images
            sample_info['output_images'] = output_images

            #print(f"Input shape: {input_images.shape}, Output shape: {output_images.shape}")
            return sample_info
               
        elif self.mode == 'test_MDvsFA':
            img_dir = os.path.join(self.imageset_dir, self.image_filenames[idx])
            gt_dir = os.path.join(self.imageset_gt_dir, self.gt_filenames[idx])

            real_input = cv2.imread(img_dir)
            real_input = cv2.resize(real_input, self.target_size)  # 调整大小
            real_input = np.float32(real_input) / 255.0

            if  ReadColorImage == 0:
                input_images = real_input * 2 - 1
            else:
                input_images = real_input[:,:,2] * 2 - 1
                input_images = np.expand_dims(input_images,axis=0)
            
            bufImg = cv2.imread(gt_dir, -1)
            dilated_bufImg = cv2.resize(bufImg,self.target_size)
            output_images = np.float32(dilated_bufImg)/255.0 # 像素归一化
            output_images = np.expand_dims(output_images,axis=0)

            sample_info = {}
            sample_info['input_images'] = input_images
            sample_info['output_images'] = output_images

            #print(f"Input shape: {input_images.shape}, Output shape: {output_images.shape}")
            return sample_info
        elif self.mode == 'test_SIRST':
            img_dir = os.path.join(self.imageset_dir, self.image_filenames[idx])
            gt_dir = os.path.join(self.imageset_gt_dir, self.gt_filenames[idx])

            real_input = cv2.imread(img_dir)
            real_input = cv2.resize(real_input, self.target_size)  # 调整大小
            real_input = np.float32(real_input) / 255.0

            if  ReadColorImage == 0:
                input_images = real_input * 2 - 1
            else:
                input_images = real_input[:,:,2] * 2 - 1
                input_images = np.expand_dims(input_images,axis=0)
            
            bufImg = cv2.imread(gt_dir, -1)
            dilated_bufImg = cv2.resize(bufImg,self.target_size)
            output_images = np.float32(dilated_bufImg)/255.0 # 像素归一化
            output_images = np.expand_dims(output_images,axis=0)

            sample_info = {}
            sample_info['input_images'] = input_images
            sample_info['output_images'] = output_images

            #print(f"Input shape: {input_images.shape}, Output shape: {output_images.shape}")
            return sample_info


class BasicBlock(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
        base_width = 64, dilation = 1, norm_layer = None):
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample= None,
        groups = 1, base_width = 64, dilation = 1, norm_layer = None,):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,block, layers,num_classes = 1000, zero_init_residual = False, groups = 1,
        width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 2
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = stride
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,  planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward(self, x) :
        return self._forward_impl(x)
    def _resnet(block, layers, pretrained_path = None, **kwargs,):
        model = ResNet(block, layers, **kwargs)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path),  strict=False)
        return model
    
    def resnet50(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 6, 3],pretrained_path,**kwargs)
    
    def resnet101(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 23, 3],pretrained_path,**kwargs)


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()

        # [b, c', h, w]
        query = self.ConvQuery(x)
        # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h).permute(0, 2, 1)
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w).permute(0, 2, 1)
        
        # [b, c', h, w]
        key = self.ConvKey(x)
        # [b, w, c', h] -> [b*w, c', h]
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h)
        # [b, h, c', w] -> [b*h, c', w]
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w)
        
        # [b, c, h, w]
        value = self.ConvValue(x)
        # [b, w, c, h] -> [b*w, c, h]
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b*w, -1, h)
        # [b, h, c, w] -> [b*h, c, w]
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b*h, -1, w)
        
        # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # [b, h, w, h+w]  concate channels in axis=3 

        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))
        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        attention_H = concate[:,:,:, 0:h].permute(0, 2, 1, 3).contiguous().view(b*w, h, h)
        attention_W = concate[:,:,:, h:h+w].contiguous().view(b*h, w, w)
 
        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h]
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma*(out_H + out_W) + x

class RCCAModule(nn.Module):
    def __init__(self, recurrence = 2, in_channels = 2048, num_classes=1):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.CCA = CrissCrossAttention(self.inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False)
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.in_channels+self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Conv2d(self.inter_channels, self.num_classes, 1)
        )

    def forward(self, x):
        # reduce channels from C to C'   2048->512
        output = self.conv_in(x)

        for i in range(self.recurrence):
            output = self.CCA(output)

        output = self.conv_out(output)
        output = self.cls_seg(torch.cat([x, output], 1))
        return output

class CCNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CCNet, self).__init__()
        self.backbone = ResNet.resnet50(replace_stride_with_dilation=[1,2,4])
        self.decode_head = RCCAModule(recurrence=2, in_channels=2048, num_classes=num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x

class Generator2_UCAN64(nn.Module):
    def __init__(self):
        super(Generator2_UCAN64, self).__init__()
        chn = 64

        # 使用更少的激活函数和卷积层
        self.leakyrelu = nn.LeakyReLU(0.2)

        # 减少膨胀卷积的层数
        self.g2_conv1 = nn.Conv2d(1, chn, 3, dilation=1, padding=1)
        self.g2_conv2 = nn.Conv2d(chn, chn, 3, dilation=2, padding=2)
        self.g2_conv3 = nn.Conv2d(chn, chn, 3, dilation=4, padding=4)
        self.g2_conv4 = nn.Conv2d(chn, chn, 3, dilation=8, padding=8)
        self.g2_conv5 = nn.Conv2d(chn, chn, 3, dilation=16, padding=16)
        self.g2_conv6 = nn.Conv2d(chn, chn, 3, dilation=32, padding=32)
        self.g2_conv7 = nn.Conv2d(chn, chn, 3, dilation=16, padding=16)  # 修改为更小的膨胀因子
        self.g2_conv8 = nn.Conv2d(chn, chn, 3, dilation=8, padding=8)    # 继续减少膨胀因子
        
        # 合并特征图的层数
        self.g2_conv9 = nn.Conv2d(chn*2, chn, 3, dilation=4, padding=4)
        self.g2_conv10 = nn.Conv2d(chn*2, chn, 3, dilation=2, padding=2)

        # 输出层
        self.g2_conv14 = nn.Conv2d(chn, 1, 1, dilation=1)

        # 合并批量归一化
        self.g2_bn = nn.BatchNorm2d(chn)

    def forward(self, input_images):  # 输入[B, 1, 128, 128],输出[B, 1, 128, 128]
        # 第一层卷积
        net1 = self.g2_conv1(input_images)
        net1 = self.g2_bn(net1)
        net1 = self.leakyrelu(net1)

        # 后续卷积层
        net2 = self.g2_conv2(net1)
        net2 = self.g2_bn(net2)
        net2 = self.leakyrelu(net2)

        net3 = self.g2_conv3(net2)
        net3 = self.g2_bn(net3)
        net3 = self.leakyrelu(net3)

        net4 = self.g2_conv4(net3)
        net4 = self.g2_bn(net4)
        net4 = self.leakyrelu(net4)

        net5 = self.g2_conv5(net4)
        net5 = self.g2_bn(net5)
        net5 = self.leakyrelu(net5)

        net6 = self.g2_conv6(net5)
        net6 = self.g2_bn(net6)
        net6 = self.leakyrelu(net6)

        # 膨胀卷积与特征融合（只保留关键部分）
        net7 = self.g2_conv7(net6)
        net7 = self.g2_bn(net7)
        net7 = self.leakyrelu(net7)

        net8 = self.g2_conv8(net7)
        net8 = self.g2_bn(net8)
        net8 = self.leakyrelu(net8)

        net9 = torch.cat([net6, net8], dim=1)  # 融合较大特征
        net9 = self.g2_conv9(net9)
        net9 = self.g2_bn(net9)
        net9 = self.leakyrelu(net9)

        net10 = torch.cat([net5, net9], dim=1)  # 融合较小特征
        net10 = self.g2_conv10(net10)
        net10 = self.g2_bn(net10)
        net10 = self.leakyrelu(net10)

        # 最后输出
        output = self.g2_conv14(net10)

        return output

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.Tanh1 = nn.Tanh()
        self.Tanh2 = nn.Tanh()
        self.Softmax = nn.Softmax()


        self.d_conv1 = nn.Conv2d(2,  24, 3, dilation=1, padding=1)
        self.d_conv2 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv3 = nn.Conv2d(24, 24, 3, dilation=1, padding=1)
        self.d_conv4 = nn.Conv2d(24, 1,  3, dilation=1, padding=1)

        self.d_bn1 = nn.BatchNorm2d(24)
        self.d_bn2 = nn.BatchNorm2d(24)
        self.d_bn3 = nn.BatchNorm2d(24)
        self.d_bn4 = nn.BatchNorm2d(1)
        self.d_bn5 = nn.BatchNorm2d(128)
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_bn7 = nn.BatchNorm2d(3)
        
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)


    def forward(self, input_images): # 输入[3B, 2, 128, 128],输出[B, 1, 128, 128]
        net = F.max_pool2d(input_images, kernel_size=[2, 2])  # [3B, 2, 64, 64]
        net = F.max_pool2d(net, kernel_size=[2, 2])  # [3B, 2, 32, 32]
        
        net = self.d_conv1(net)
        net = self.d_bn1(net)
        net = self.leakyrelu1(net)

        net = self.d_conv2(net)
        net = self.d_bn2(net)
        net = self.leakyrelu2(net)

        net = self.d_conv3(net)
        net = self.d_bn3(net)
        net = self.leakyrelu3(net)

        net = self.d_conv4(net)
        net = self.d_bn4(net)
        net1 = self.leakyrelu4(net) # [3B, 1, 32, 32]

        net = net1.view(-1, 1024) # [3B, 1024]
        net = self.fc1(net)      # [3B, 128]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn5(net)
        net = self.Tanh1(net)    # [3B, 128, 1, 1]

        net = net.view(-1, 128) # [3B, 128]
        net = self.fc2(net)      # [3B, 64]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn6(net)
        net = self.Tanh2(net)   # [3B, 64, 1, 1]

        net = net.view(-1, 64) # [3B, 64]
        net = self.fc3(net)      # [3B, 3]
        net = net.unsqueeze(2).unsqueeze(3)
        net = self.d_bn7(net)
        net = self.Softmax(net) # [3B, 3, 1, 1]
        net = net.squeeze(3).squeeze(2)
        
        realscore0, realscore1, realscore2 = torch.split(net, mini_batch_size, dim=0)
        feat0, feat1, feat2 = torch.split(net1, mini_batch_size, dim=0)
        featDist = torch.mean(torch.pow(feat1 - feat2, 2))

        return realscore0, realscore1, realscore2, featDist
8
if __name__ == '__main__':
    f1_values = []  # 用于存储每十个 batch 的 F1 值
    batch_indices = []  # 用于存储对应的 batch 索引

    # 保存输出的总路径
    root_result_dir = os.path.join('pytorch_outputs') 
    os.makedirs(root_result_dir, exist_ok=True)
    root_result_models_dir = os.path.join(root_result_dir,'models')
    os.makedirs(root_result_models_dir, exist_ok=True)
    root_result_image_dir = os.path.join(root_result_dir,'results')
    os.makedirs(root_result_image_dir, exist_ok=True)

    # 日志文件的目录
    log_dir = os.path.join(root_result_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

    # 当前时间，日志文件的后缀
    time_suffix = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    # 日志文件
    log_file = os.path.join(log_dir, 'log_train_g1g2_{}.txt'.format(time_suffix))
    logger = create_logger(log_file)

    # 定义dataset
    trainsplit = G1G2Dataset(mode='train')
    trainset = DataLoader(trainsplit, batch_size=mini_batch_size, pin_memory=True,
                              num_workers=4, shuffle=True, drop_last=True)
    testsplit = G1G2Dataset(mode='test_MDvsFA')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                              num_workers=4, shuffle=False, drop_last=True)
    testsplit2 = G1G2Dataset(mode='test_SIRST')
    testset2 = DataLoader(testsplit, batch_size=1, pin_memory=True,
                              num_workers=4, shuffle=False, drop_last=True)
    
    # 定义3个Model
    g1 = CCNet()
    g1.cuda()
    g2 = Generator2_UCAN64()
    g2.cuda()
    dis = discriminator()
    dis.cuda()

    # 定义3个优化器
    optimizer_g1 = optim.Adam(g1.parameters(), lr=1e-4, betas=(0.5,0.999))
    optimizer_g2 = optim.Adam(g2.parameters(), lr=1e-4, betas=(0.5,0.999))
    optimizer_d = optim.Adam(dis.parameters(), lr=1e-5, betas=(0.5,0.999))

    # 定义loss
    loss1 = nn.BCEWithLogitsLoss()

    it = 0
    
    for epoch in range(0, max_epoch_num):
        # 调整学习率
        if (epoch+1) % 10 == 0:
            for p in optimizer_g1.param_groups:
                p['lr'] *= 0.2
            for q in optimizer_g2.param_groups:
                q['lr'] *= 0.2
            for r in optimizer_g2.param_groups:
                r['lr'] *= 0.2
        # 训练一个周期
        logger.info('Now we are training epoch {}!'.format(epoch+1))
        total_it_per_epoch = len(trainset)
        for bt_idx, data in enumerate(trainset):
            # 训练一个batch
            torch.cuda.empty_cache() # 释放之前占用的缓存
            it = it + 1
            logger.info('current iteration {}/{}, epoch {}/{}, total iteration: {}, g1 lr: {}, g2 lr: {}, Dis lr: {}'.format(
            bt_idx+1, total_it_per_epoch, epoch+1, max_epoch_num, it, float(optimizer_g1.param_groups[0]['lr']), 
            float(optimizer_g2.param_groups[0]['lr']), float(optimizer_d.param_groups[0]['lr'])))

            # 先训练判别器
            dis.train() 
            g1.eval()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            # 将输入输出放到cuda上
            input_images, output_images = data['input_images'], data['output_images']   # [B, 1, 128, 128]
            input_images = input_images.cuda(non_blocking=True).float()
            output_images = output_images.cuda(non_blocking=True).float()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            ES0 = torch.mean(loss1(logits_real, gen_gt)) 
            ES1 = torch.mean(loss1(logits_fake1, gen_gt1)) 
            ES2 = torch.mean(loss1(logits_fake2, gen_gt2))

            disc_loss = ES0 + ES1 + ES2
            logger.info(" discriminator loss is {}".format(disc_loss))
            disc_loss.backward()  # 将误差反向传播
            optimizer_d.step()  # 更新参数

            # 再训练g1
            dis.eval() 
            g1.train()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)
            MD1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), output_images))
            FA1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), 1 - output_images))
            MF_loss1 = lambda1 * MD1 + FA1 

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)


            gen_adv_loss1 = torch.mean(loss1(logits_fake1, gen_gt))
            gen_loss1  = 100*MF_loss1 + 10*gen_adv_loss1 + 1*Lgc
            logger.info(" g1 loss is {}".format(gen_loss1))

            gen_loss1.backward()  # 将误差反向传播
            optimizer_g1.step()  # 更新参数

            # 再训练g2
            dis.eval() 
            g1.eval()
            g2.train()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)
            MD2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), output_images))
            FA2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), 1 - output_images))
            MF_loss2 = MD2 + lambda2 * FA2

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)


            gen_adv_loss2 = torch.mean(loss1(logits_fake2, gen_gt)) 
            gen_loss2  = 100*MF_loss2 + 10*gen_adv_loss2 + 1*Lgc
            logger.info(" g2 loss is {}".format(gen_loss2))

            gen_loss2.backward()  # 将误差反向传播
            optimizer_g2.step()  # 更新参数

            if (bt_idx+1) % 10 == 0:
                # 在测试集上测试
                sum_val_loss_g1 = 0
                sum_val_false_ratio_g1 = 0 
                sum_val_detect_ratio_g1 = 0
                sumRealTarN_g1 = 0
                sumDetTarN_g1 = 0
                sum_val_F1_g1 = 0
                g1_time = 0

                sum_val_loss_g2 = 0
                sum_val_false_ratio_g2 = 0 
                sum_val_detect_ratio_g2 = 0
                sumRealTarN_g2 = 0
                sumDetTarN_g2 = 0
                sum_val_F1_g2 = 0
                g2_time = 0

                sum_val_loss_g3 = 0
                sum_val_false_ratio_g3 = 0 
                sum_val_detect_ratio_g3 = 0
                sumRealTarN_g3 = 0
                sumDetTarN_g3 = 0
                sum_val_F1_g3 = 0

                for bt_idx_test, data in enumerate(testset):
                    g1.eval()
                    g2.eval()
                    dis.eval()
                    optimizer_g1.zero_grad()
                    optimizer_g2.zero_grad()
                    optimizer_d.zero_grad()

                    # 将输入输出放到cuda上
                    input_images, output_images = data['input_images'], data['output_images']   # [B, 1, 128, 128]
                    input_images = input_images.cuda(non_blocking=True).float()
                    output_images = output_images.cuda(non_blocking=True).float()

                    stime = time.time()
                    g1_out = g1(input_images) # [B, 1, 128, 128]
                    etime = time.time()
                    g1_time += etime - stime
                    #logger.info('testing {}, g1 time is {}'.format(bt_idx_test, etime-stime))
                    g1_out = torch.clamp(g1_out, 0.0, 1.0)

                    stime = time.time()
                    g2_out = g2(input_images) # [B, 1, 128, 128]
                    etime = time.time()
                    g2_time += etime - stime
                    #logger.info('testing {}, g2 time is {}'.format(bt_idx_test, etime-stime))
                    g2_out = torch.clamp(g2_out, 0.0, 1.0)

                    g3_out = (g1_out + g2_out) / 2 # 取均值的方式进行融合

                    output_images = output_images.cpu().numpy()
                    g1_out = g1_out.detach().cpu().numpy()
                    g2_out = g2_out.detach().cpu().numpy()
                    g3_out = g3_out.detach().cpu().numpy()
                    # 算g1
                    val_loss_g1 = np.mean(np.square(g1_out - output_images))
                    sum_val_loss_g1 += val_loss_g1
                    val_false_ratio_g1 = np.mean(np.maximum(0, g1_out - output_images))
                    sum_val_false_ratio_g1 += val_false_ratio_g1
                    val_detect_ratio_g1 = np.sum(g1_out * output_images)/np.maximum(np.sum(output_images),1)
                    sum_val_detect_ratio_g1 += val_detect_ratio_g1
                    val_F1_g1 = calculateF1Measure(g1_out, output_images, 0.5)
                    sum_val_F1_g1 += val_F1_g1

                    # 算g2
                    val_loss_g2 = np.mean(np.square(g2_out - output_images))
                    sum_val_loss_g2 += val_loss_g2
                    val_false_ratio_g2 = np.mean(np.maximum(0, g2_out - output_images))
                    sum_val_false_ratio_g2 += val_false_ratio_g2
                    val_detect_ratio_g2 = np.sum(g2_out * output_images)/np.maximum(np.sum(output_images),1)
                    sum_val_detect_ratio_g2 += val_detect_ratio_g2
                    val_F1_g2 = calculateF1Measure(g2_out, output_images, 0.5)
                    sum_val_F1_g2 += val_F1_g2

                    # 算g3
                    val_loss_g3 = np.mean(np.square(g3_out - output_images))
                    sum_val_loss_g3 += val_loss_g3
                    val_false_ratio_g3 = np.mean(np.maximum(0, g3_out - output_images))
                    sum_val_false_ratio_g3 += val_false_ratio_g3
                    val_detect_ratio_g3 = np.sum(g3_out * output_images)/np.maximum(np.sum(output_images),1)
                    sum_val_detect_ratio_g3 += val_detect_ratio_g3
                    val_F1_g3 = calculateF1Measure(g3_out, output_images, 0.5)
                    sum_val_F1_g3 += val_F1_g3

                    # 保存图片
                    output_image1 = np.squeeze(g1_out*255.0)#/np.maximum(output_image1.max(),0.0001))
                    output_image2 = np.squeeze(g2_out*255.0)#/np.maximum(output_image2.max(),0.0001))
                    output_image3 = np.squeeze(g3_out*255.0)#/np.maximum(output_image3.max(),0.0001))
                    #cv2.imwrite("%s/%05d_grt.png"%(task,ind),np.uint8(np.squeeze(gt_image*255.0)))
                    cv2.imwrite("pytorch_outputs/results/%05d_G1.png"%(bt_idx_test),np.uint8(output_image1))
                    cv2.imwrite("pytorch_outputs/results/%05d_G2.png"%(bt_idx_test),np.uint8(output_image2))
                    cv2.imwrite("pytorch_outputs/results/%05d_Res.png"%(bt_idx_test),np.uint8(output_image3))                

                logger.info("======================== g1 results ============================")
                avg_val_loss_g1 = sum_val_loss_g1/100
                avg_val_false_ratio_g1  = sum_val_false_ratio_g1/100
                avg_val_detect_ratio_g1 = sum_val_detect_ratio_g1/100
                avg_val_F1_g1 = sum_val_F1_g1/100

                logger.info("================val_L2_loss is %f"% (avg_val_loss_g1))
                logger.info("================falseAlarm_rate is %f"% (avg_val_false_ratio_g1))
                logger.info("================detection_rate is %f"% (avg_val_detect_ratio_g1))
                logger.info("================F1 measure is %f"% (avg_val_F1_g1))
                logger.info("g1 time is {}".format(g1_time))

                logger.info("======================== g2 results ============================")
                avg_val_loss_g2 = sum_val_loss_g2/100
                avg_val_false_ratio_g2  = sum_val_false_ratio_g2/100
                avg_val_detect_ratio_g2 = sum_val_detect_ratio_g2/100
                avg_val_F1_g2 = sum_val_F1_g2/100

                logger.info("================val_L2_loss is %f"% (avg_val_loss_g2))
                logger.info("================falseAlarm_rate is %f"% (avg_val_false_ratio_g2))
                logger.info("================detection_rate is %f"% (avg_val_detect_ratio_g2))
                logger.info("================F1 measure is %f"% (avg_val_F1_g2))
                logger.info("g2 time is {}".format(g2_time))

                logger.info("======================== g3 results ============================")
                avg_val_loss_g3 = sum_val_loss_g3/100
                avg_val_false_ratio_g3  = sum_val_false_ratio_g3/100
                avg_val_detect_ratio_g3 = sum_val_detect_ratio_g3/100
                avg_val_F1_g3 = sum_val_F1_g3/100
                f1_values.append(avg_val_F1_g3)  # 记录 F1 值
                batch_indices.append(it + 1)  # 记录 batch 索引

                logger.info("================val_L2_loss is %f"% (avg_val_loss_g3))
                logger.info("================falseAlarm_rate is %f"% (avg_val_false_ratio_g3))
                logger.info("================detection_rate is %f"% (avg_val_detect_ratio_g3))
                logger.info("================F1 measure is %f"% (avg_val_F1_g3))

                ############# save model
                #ckpt_name1 = os.path.join(root_result_dir, 'models/g1_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                #ckpt_name2 = os.path.join(root_result_dir, 'models/g2_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                #ckpt_name3 = os.path.join(root_result_dir, 'models/dis_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                #save_checkpoint(checkpoint_state(g1, optimizer_g1, epoch+1, it), filename=ckpt_name1)
                #save_checkpoint(checkpoint_state(g2, optimizer_g2, epoch+1, it), filename=ckpt_name2)
                #save_checkpoint(checkpoint_state(dis, optimizer_d, epoch+1, it), filename=ckpt_name3)

        if (epoch + 1) % 1 == 0:  # 每个 epoch 结束时保存模型
            # 保存模型
            ckpt_name1 = os.path.join(root_result_dir, 'models/g1_epoch_{}_batch_{}'.format(epoch + 1, 'final'))
            ckpt_name2 = os.path.join(root_result_dir, 'models/g2_epoch_{}_batch_{}'.format(epoch + 1, 'final'))
            ckpt_name3 = os.path.join(root_result_dir, 'models/dis_epoch_{}_batch_{}'.format(epoch + 1, 'final'))
            save_checkpoint(checkpoint_state(g1, optimizer_g1, epoch + 1, it), filename=ckpt_name1)
            save_checkpoint(checkpoint_state(g2, optimizer_g2, epoch + 1, it), filename=ckpt_name2)
            save_checkpoint(checkpoint_state(dis, optimizer_d, epoch + 1, it), filename=ckpt_name3)

         # 可视化 F1 值
    plt.plot(batch_indices, f1_values)
    plt.xlabel('Batch')
    plt.ylabel('F1 Value')
    plt.grid()
    plt.savefig('f1_values.png')  # 保存图像
    plt.show()  # 显示图像