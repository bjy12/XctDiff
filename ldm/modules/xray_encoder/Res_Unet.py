import pdb
import copy
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
import pdb
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #pdb.set_trace()
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
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def get_data_transform(in_channels, input_img_size):
    data_aug = []
    data_aug += [torchvision.transforms.Resize(input_img_size)]
    if in_channels == 1:
        data_aug += [torchvision.transforms.Normalize([0.485], [0.229])]
    elif in_channels == 3:
        data_aug += [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    data_aug = torchvision.transforms.Compose(data_aug)

    return data_aug        
def set_model(pretrained_weight, weight_dir, model_name):
    print(f"model name : {model_name}")
    if pretrained_weight == "imagenet":
        model = getattr(torchvision.models, model_name)(pretrained=True)
    # elif pretrained_weight == "autoenc_LIDC":
    #     model = getattr(torchvision.models, model_name)(pretrained=False)
    #     pretrained_weight = torch.load(weight_dir)
    #     model = update_pretrained_weight(model, pretrained_weight["network_state_dict"])
    elif pretrained_weight is None:
        model = getattr(torchvision.models, model_name)(pretrained=False)
        print("Use RenNet34 Image Encoder from the scratch")
    else:
        raise NotImplementedError
    return model


class ResNetGLEncoder(nn.Module):
    def __init__(self, in_channels ,input_img_size , encoder_freeze_layer,
                       feature_layer , global_feature_layer , global_feature_layer_last,pretrained ,
                       weight_dir , bilinear , n_classed ,
                       model_name ):
        super(ResNetGLEncoder, self).__init__()
        self.in_channels = in_channels
        self.input_img_size = input_img_size
        self.encoder_freeze_layer = encoder_freeze_layer
        self.feature_layer =feature_layer   ## for local
        self.global_feature_layer = global_feature_layer  ## for local
        self.global_feature_layer_last = global_feature_layer_last
        self.pretrained = pretrained
        self.weight_dir = weight_dir
        self.bilinear = bilinear
        self.n_classed = n_classed
        gloabl_out_dim = 128
        model_name = model_name 

        assert not ((self.in_channels == 1) and self.encoder_freeze_layer)
        assert self.feature_layer in ["layer1", "layer2", "layer3", "layer4", "all"]    ## layer2
        assert self.pretrained in [None, "autoenc_LIDC", "imagenet"]                    ## imagenet
        self.global_feature_layers_in_last = ['conv1', 'bn1', 'relu']
        self.model = set_model(pretrained_weight=self.pretrained, weight_dir=self.weight_dir, model_name=model_name)

        del self.model.fc
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.in_channels != 3:
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # delete layers after feature_layer
        for layer in ["layer1", "layer2", "layer3", "layer4"][::-1]:
            if layer == self.global_feature_layer:
                break
            delattr(self.model, layer)
        #pdb.set_trace()
        if self.global_feature_layer_last is not None:
            tmp = getattr(self.model, self.global_feature_layer)
            setattr(self.model, self.global_feature_layer, tmp[:self.global_feature_layer_last+1])
            tmp = getattr(self.model, self.global_feature_layer)[-1]
            new_tmp = []
            for name, layer in tmp.named_modules():
                if len(name) > 0 and name in  self.global_feature_layers_in_last:
                    new_tmp.append(copy.deepcopy(layer))
            setattr(getattr(self.model, self.global_feature_layer), str(self.global_feature_layer_last), nn.Sequential(*new_tmp))
        #pdb.set_trace()

        ## get feature_layer's dim
        tmp = [n for n in getattr(self.model, self.feature_layer)[-1].named_modules() if len(n[0]) > 0]
        #pdb.set_trace()
        for name, layer in tmp[::-1]:
            if name[:4] == 'conv':
                output_dim = layer.out_channels
                break
        
        tmp = [n for n in getattr(self.model, self.global_feature_layer)[-1].named_modules() if len(n[0]) > 0]
        for name, layer in tmp[::-1]:
            if isinstance(layer, nn.Conv2d):
                self.global_output_dim = layer.out_channels
                break
        #pdb.set_trace()
        # if output_dim != self.latent_dim:
        #     self.conv2 = nn.Conv2d(output_dim, self.latent_dim, kernel_size=(3, 3), padding=1, bias=False)  ## batch x latent_dim x 10 x 10
        #     self.bn2 = nn.BatchNorm2d(self.latent_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     self.relu = nn.ReLU()

        if self.pretrained and self.encoder_freeze_layer:
            for name, param in self.model.named_parameters():
                if self.encoder_freeze_layer in name:
                    break
                param.requires_grad = False
        factor = 2 if self.bilinear else 1
        # Decoder部分，使用新的Up模块
        #self.up3 = Up(1024, 512 // factor, self.bilinear)  # layer3 -> layer2
        self.up2 = Up(512, 128 // factor, self.bilinear)   # layer2 -> layer1
        self.up1 = Up(128, 64, self.bilinear)    # layer1 -> initial

        self.last_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

       # 在获取到原始的 global_output_dim 后添加转换层
        self.global_transform = nn.Sequential(
            nn.Conv2d(self.global_output_dim, gloabl_out_dim, kernel_size=1),
            nn.BatchNorm2d(gloabl_out_dim),
        )

        self.outc = nn.Sequential(OutConv(64 ,self.n_classed) ,
                                  nn.BatchNorm2d(self.n_classed) ,
                                  nn.Tanh())

        # self.output_dim = output_dim
        # self.output_ch = self.latent_dim

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_dim)
        """
        #pdb.set_trace()    
        if self.in_channels == 3 and x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)  # 1 3 256 256 
        skip_connections = {}
        # x = self.transform(x)
        # pdb.set_trace()
        x = self.model.conv1(x)   # 1 64 128 128 
        x = self.model.bn1(x)     # 1 64 128 128
        x = self.model.relu(x)    # 1 64 128 128 
        skip_connections['initial'] = x 
        x = self.model.maxpool(x)  # 1  64  64 64  
        #pdb.set_trace()
        # layer1: 256 64 64 ,layer2: 512 32 32,layer3:(global_feature  1 , 256 ,1,1 )
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
        #for layer in ["layer1" , "layer2", "layer3"]:
            x = getattr(self.model, layer)(x)
            skip_connections[layer] = x
            if layer == self.feature_layer:
                local_feature = x
            if layer == self.global_feature_layer:
                #pdb.set_trace()
                global_feature = self.model.avgpool(x)
                global_feature = self.global_transform(global_feature)
                #pdb.set_trace()
                break
        #pdb.set_trace()
        
        # Decoder path with skip connections
        #local_feature = self.up3(local_feature, skip_connections['layer2'])
        #pdb.set_trace()
        # 512 32 32  skips 256 64 64 
        
        multi_level_features = []

        local_feature = self.up2(local_feature, skip_connections['layer1'])  # 32
        #pdb.set_trace()
        level_3_feature = self.up1(local_feature, skip_connections['initial']) # 64
        #pdb.set_trace()
        level_2_feature = self.last_up(level_3_feature) # 128
        level_1_feature = self.outc(level_2_feature) # 256
        #pdb.set_trace()

        return level_1_feature.float() , global_feature.float()

if __name__ == "__main__":
  
    # 初始化参数，使用关键字参数方式
    encoder = ResNetGLEncoder(
        in_channels=3,
        latent_dim=512,
        input_img_size=256,
        encoder_freeze_layer='layer1',
        feature_layer="layer2",
        global_feature_layer="layer3",
        global_feature_layer_last=22,
        pretrained="imagenet",
        weight_dir='',
        bilinear=False,
        n_classed=64,
        model_name="resnet101"
    )
    
    # 移动模型到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    pdb.set_trace()
    x = torch.randn(1, 1, 256, 256).to('cuda')
    out = encoder(x)
    print(out['local'].shape, out['global'].shape)
