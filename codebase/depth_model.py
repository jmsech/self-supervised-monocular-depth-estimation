import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from collections import OrderedDict


############## Utils to Build Model Layers ########################

def load_resnet():
    """
    input dimes: (bs, depth=3, H, W)
    output dims: (bs, depth=512, H/128, W/128)
    """

    # load pre-trained ResNet
    resnet = resnet18(pretrained=True)

    # drop AdaptiveAvgPool2d and Linear layers (last two)
    named_layers = OrderedDict(list(resnet.named_children())[:-2])

    return nn.Sequential(named_layers)


def decoder_block(in_dim, out_dim, kernel=3, stride=1, pad=1):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel, stride, pad),
                         nn.BatchNorm2d(out_dim),
                         nn.ELU()
                         )

def calc_disp(in_dim):
    # 2 layers: one for Left, one for Right disparity
    # potentiall can ad BatchNormalization here
    return nn.Sequential(
        nn.Conv2d(in_dim, 2, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    )


############################ Model ###############################

class MonoDepthModel(nn.Module):
    def __init__(self):
        super().__init__()

        ### Encoder ###
        resnet = load_resnet()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.maxpool = resnet.maxpool

        self.resnet1 = resnet.layer1
        self.resnet2 = resnet.layer2
        self.resnet3 = resnet.layer3
        self.resnet4 = resnet.layer4

        ### Decoder ###
        self.updisp = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        disp_dim = 2  # Left and Right

        self.upsample6 = nn.ConvTranspose2d(512, 512, 3, stride=2,  padding=1)  # H/64 -> H/32 double the Height/Width at each stage
        self.bn512 = nn.BatchNorm2d(512)
        self.deconv6 = decoder_block(512 + 256, 512, 3, 1)

        self.upsample5 = nn.ConvTranspose2d(512, 256, 3, 2, padding=1)  # -> H/16
        self.bn256 = nn.BatchNorm2d(256)
        self.deconv5 = decoder_block(256 + 128, 256, 3, 1)

        self.upsample4 = nn.ConvTranspose2d(256, 128, 3, 2, 1)  # -> H/8
        self.bn128 = nn.BatchNorm2d(128)
        self.deconv4 = decoder_block(128 + 64, 128, 3, 1)
        self.calc_disp4 = calc_disp(128)

        self.upsample3 = nn.ConvTranspose2d(128, 64, 3, 1, 1)  # -> H/4
        self.bn64 = nn.BatchNorm2d(64)
        self.deconv3 = decoder_block(64 + 64 + disp_dim, 64, 3, 1)
        self.calc_disp3 = calc_disp(64)

        self.upsample2 = nn.ConvTranspose2d(64, 32, 3, 2, 1)  # -> H/2
        self.bn32 = nn.BatchNorm2d(32)
        self.deconv2 = decoder_block(32 + 64 + disp_dim, 32, 3, 1)
        self.calc_disp2 = calc_disp(32)

        self.upsample1 = nn.ConvTranspose2d(32, 16, 3, 2, 1)  # -> H
        self.bn16 = nn.BatchNorm2d(16)
        self.deconv1 = decoder_block(16 + disp_dim, 16, 3, 1)
        self.final_disp = calc_disp(16)

    def forward(self, input):
        #print("Input:    ", input.shape)
        ### ENCODER ###
        x = self.conv1(input)
        skip1 = x
        #print("Skip1", skip1.shape)
        #print()

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        skip2 = x
        #print("Skip2:", skip2.shape)
        #print()

        x = self.resnet1(x)
        skip3 = x
        #print("Skip3:", skip3.shape)
        #print()

        x = self.resnet2(x)
        skip4 = x
        #print("Skip4:", skip4.shape)
        #print()

        x = self.resnet3(x)
        skip5 = x
        #print("Skip5:", skip5.shape)
        #print()

        x = self.resnet4(x)
        #print("Encoded x:", x.shape, "\n")

        ### DECODER ###
        # UpConv6
        x = self.upsample6(x, output_size=skip5.size())
        x = self.bn512(x)
        x = self.elu(x)
        #print("UpConv6:  ", x.shape, " Skip5:", skip5.shape)
        x = torch.cat((x, skip5), dim=1)
        #print("Cat x:    ", x.shape)
        x = self.deconv6(x)
        #print("Deonv x:  ", x.shape, "\n")

        # UpConv5
        x = self.upsample5(x, output_size=skip4.size())
        x = self.bn256(x)
        x = self.elu(x)
        #print("UpConv5:  ", x.shape, " Skip4:", skip4.shape)
        x = torch.cat((x, skip4), dim=1)
        #print("Cat x:    ", x.shape)
        x = self.deconv5(x)
        #print("Deonv x:  ", x.shape, "\n")

        # UpConv4 + Disp4
        x = self.upsample4(x, output_size=skip3.size())
        x = self.bn128(x)
        x = self.elu(x)
        #print("UpConv4:  ", x.shape, " Skip3:", skip3.shape)
        x = torch.cat((x, skip3), dim=1)
        #print("Cat x:    ", x.shape)
        x = self.deconv4(x)
        #print("Deonv x:  ", x.shape, "\n")

        # this disparity is correct size for concatentaion but need to be halved for the model output
        self.disp4 = 0.3 * self.calc_disp4(x)
        #print("Disp4:    ", self.disp4.shape)
        updisp4 = self.disp4
        #print("UpDisp4:  ", updisp4.shape, "\n")
        self.disp4 = F.interpolate(self.disp4, scale_factor=0.5, mode="bilinear", align_corners=True)

        # ----------Concat disparity----

        # UpConv3 + Disp3
        x = self.upsample3(x , output_size=skip2.size())
        x = self.bn64(x)
        x = self.elu(x)
        #print("UpConv3:  ", x.shape, " Skip2:", skip2.shape, " UpDisp4:", updisp4.shape)
        x = torch.cat((x, skip2, updisp4), dim=1)
        #print("Cat x:    ", x.shape)
        x = self.deconv3(x)
        #print("Deonv x:  ", x.shape, "\n")

        self.disp3 = 0.3 * self.calc_disp3(x)
        updisp3 = self.updisp(self.disp3)

        # UpConv2 + Disp2
        x = self.upsample2(x, output_size=skip1.size())
        x = self.bn32(x)
        x = self.elu(x)
        #print("UpConv2:  ", x.shape, " Skip1:", skip1.shape, " UpDisp3:", updisp3.shape)
        x = torch.cat((x, skip1, updisp3), dim=1)
        #print("Cat x:    ", x.shape)
        x = self.deconv2(x)
        #print("Deonv x:  ", x.shape, "\n")

        self.disp2 = 0.3 * self.calc_disp2(x)
        updisp2 = self.updisp(self.disp2)

        # UpConv1 + FinalDisp
        x = self.upsample1(x, output_size=updisp2.size())
        x = self.bn16(x)
        x = self.elu(x)
        #print("UpConv1:  ", x.shape, " UpDisp2:", updisp2.shape)
        x = torch.cat((x, updisp2), dim=1)
        #print("Cat x:    ", x.shape)
        x = self.deconv1(x)
        #print("Deonv x:  ", x.shape, "\n")

        self.disp1 = 0.3 * self.final_disp(x)

        #print("Disp1:", self.disp1.shape, "Disp2:", self.disp2.shape, "Disp2:", self.disp3.shape, "Disp4:", self.disp4.shape, "\n")
        return self.disp1, self.disp2, self.disp3, self.disp4
