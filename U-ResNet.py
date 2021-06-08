import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import torchvision.models as models

# Weights initializer
def weights_init_normal(m):
    """
    Initilize weights with normal distribution to the networks
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Pretrain VGG19
class FeatureExtractor(nn.Module):
    """
    This is a vgg19 pretrained model as a feature extractor which outpu features of last convolution layer
    which goes to the content loss function
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # Extracting features at different level
        self.feature_extractor1 = nn.Sequential(
            *list(vgg19_model.features.children())[:35]
        )

    def forward(self, img):
        out1 = self.feature_extractor1(img)
        return out1


# Residual block
class ResBl(nn.Module):
    """
    This is the residual block class, this block is part of the main block (MainBl) 
    Args:
        in_size: number of input channels
        out_size: number of input channels
    """
    def __init__(self, in_size, out_size):
        super(ResBl, self).__init__()

        self.conv1 = nn.Conv2d(in_size, (in_size + out_size) // 2, 3, 1, 1, bias=True)
        self.NL1 = nn.BatchNorm2d(in_size, affine=True)

        self.conv2 = nn.Conv2d(
            (in_size + out_size) // 2, (in_size + out_size) // 2, 3, 1, 1, bias=True
        )
        self.NL2 = nn.BatchNorm2d((in_size + out_size) // 2, affine=True)

        self.conv3 = nn.Conv2d((in_size + out_size) // 2, out_size, 3, 1, 1, bias=True)
        self.NL3 = nn.BatchNorm2d((in_size + out_size) // 2, affine=True)

    def forward(self, x):
        out1 = F.relu(self.NL1((x[0], x[1])))
        out2 = self.conv1(out1)
        out3 = F.relu(self.NL2((out2, x[1])))
        out4 = self.conv2(out3)
        out5 = F.relu(self.NL3((out4, x[1])))
        out6 = self.conv3(out5)

        return out6


# Concatenation block
class Concatenation(nn.Module):
    """
    This is the Concatenation block class, this block is part of the main block (MainBl) 
    and concatenate the feature maps from previous up-sampling block and corresponding down-sampling block
    NL represents Normalization Layer
    out_n represents the output after n number of operations
    Args:
        in_size: number of input channels
        out_size: number of input channels
    """
    def __init__(self, in_size, out_size):
        super(Concatenation, self).__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, 3, 1, 1, bias=True)
        self.NL1 = nn.BatchNorm2d(out_size, affine=True)
        self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1, bias=True)

    def forward(self, x):

        out1 = self.conv1(x[0])
        out2 = F.relu(self.NL1((out1, x[1])))
        out3 = self.conv2(out2)

        return out3


class MainBl(nn.Module):
    """ 
    The implementation of the main block class which is used in down-sampling block, bridge block, and up-sampling block
    NL represents Normaization Layer
    out_n represents the output after n number of operations
    resbl is an instantiation of residual block
    Args:
        in_size: number of input channels
        out_size: number of input channels
        
    """

    def __init__(self, in_size, out_size):
        super(MainBl, self).__init__()

        self.resbl = ResBl(in_size, out_size)

        self.conv = nn.Conv2d(in_size, out_size, 3, 1, 1, bias=True)
        self.NL = nn.BatchNorm2d(in_size, affine=True)

    def forward(self, x):
        out1 = self.resbl(x)
        out2 = F.relu(self.NL((x[0], x[1])))
        out3 = self.conv(out2)
        return out1 + out3


class GeneratorUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.pooling = nn.MaxPool2d(2, stride=2)
        self.upsampling = nn.Upsample(scale_factor=2, mode="bilinear")

        self.ch1 = nn.Conv2d(1, 32, 3, 1, 1, bias=True)
        self.ch2 = nn.Conv2d(1, 64, 3, 1, 1, bias=True)
        self.ch3 = nn.Conv2d(1, 128, 3, 1, 1, bias=True)
        self.ch4 = nn.Conv2d(1, 256, 3, 1, 1, bias=True)

        self.con5 = Concatenation(512, 256)
        self.con6 = Concatenation(256, 128)
        self.con7 = Concatenation(128, 64)

        self.down1 = MainBl(32, 64)
        self.down2 = MainBl(64, 128)
        self.down3 = MainBl(128, 256)
        self.bridge = MainBl(256, 256)

        self.up1 = MainBl(256, 128)
        self.up2 = MainBl(128, 64)
        self.up3 = MainBl(64, 32)

        self.conv = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1, bias=True), nn.Tanh())
        # For segmentation task, we remove the Tanh activation function
    def forward(self, x):

        # The implementation of the UResNet consisting of three down-sampling block, a bridge block, and three up-sampling block
        # This network serves as the generator of our framwork
        # chN and conN is for matching the number of feature maps with the corresponding block
        # dN represents feature maps resulting from a block before max poooling or up-sampling
        # dN_ represents feature maps resulting from a block after max poooling or up-sampling
        # x is the input and c is the conditions
        
        d1 = self.ch1(x[0])
        d2 = self.down1((d1, x[1]))
        d2_ = self.pooling(d2)
        x_1 = self.pooling(x[0])
        c_1 = self.pooling(x[1])

        d3 = self.ch2(x_1)
        d4 = self.down2((d3 + d2_, c_1))
        d4_ = self.pooling(d4)
        x_2 = self.pooling(x_1)
        c_2 = self.pooling(c_1)

        d5 = self.ch3(x_2)
        d6 = self.down3((d5 + d4_, c_2))
        d6_ = self.pooling(d6)
        x_3 = self.pooling(x_2)
        c_3 = self.pooling(c_2)

        d7 = self.ch4(x_3)
        d8 = self.bridge((d7 + d6_, c_3))
        u1 = self.upsampling(d8)
        M1 = torch.cat((d6, u1), dim=1)

        d9 = self.con5((M1, c_2))
        d10 = self.up1((d9, c_2))
        u2 = self.upsampling(d10)
        M2 = torch.cat((d4, u2), dim=1)

        d11 = self.con6((M2, c_1))
        d12 = self.up2((d11, c_1))
        u3 = self.upsampling(d12)
        M3 = torch.cat((d2, u3), dim=1)

        d13 = self.con7((M3, x[1]))
        d14 = self.up3((d13, x[1]))

        uf = self.conv(d14)
        #

        return uf


# Discriminator
# This has been borrowed from https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = 19
        for out_filters, stride, normalize in [ (64, 1, True),
                                                (64, 2, True),
                                                (128, 1, True),
                                                (128, 2, True),
                                                (256, 1, True),
                                                (256, 2, True),
                                                (512, 1, True),
                                                (512, 2, True),]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)