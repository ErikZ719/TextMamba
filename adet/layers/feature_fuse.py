import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels=256, inter_channels=64, out_features_num=4, attention_type='scale_channel_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        inner_channels = inter_channels
        self.out_features_num = out_features_num
        bias = False

        self.output_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=bias),
                                                         nn.GroupNorm(32, in_channels)) for i in range(4)])

        self.lateral_conv_4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.lateral_conv_3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.lateral_conv_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=bias),
                                            nn.GroupNorm(32, in_channels))
        self.linear_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(ori_channels, in_channels, 1, bias=bias),
                                                        nn.GroupNorm(32, in_channels)) for ori_channels in
                                          [256, 512, 1024, 2048]])

        for i in range(4):
            self.output_convs[i].apply(self._initialize_weights)
        self.lateral_conv_4.apply(self._initialize_weights)
        self.lateral_conv_3.apply(self._initialize_weights)
        self.lateral_conv_2.apply(self._initialize_weights)
        self.fpems = nn.ModuleList()
        for i in range(4):
            self.fpems.append(FPEM(in_channels))
    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('GroupNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, encoder_features, bk_feature):
        c2, c3, c4, c5 = encoder_features

        # Basic FPN
        out4 = F.interpolate(c5, size=c4.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_4(c4) \
               + self.linear_proj[2](
            F.interpolate(bk_feature[2], size=c4.shape[-2:], mode="bilinear", align_corners=False))
        out3 = F.interpolate(out4, size=c3.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_3(c3) \
               + self.linear_proj[1](
            F.interpolate(bk_feature[1], size=c3.shape[-2:], mode="bilinear", align_corners=False))
        out2 = F.interpolate(out3, size=c2.shape[-2:], mode="bilinear", align_corners=False) + self.lateral_conv_2(c2) \
               + self.linear_proj[0](
            F.interpolate(bk_feature[0], size=c2.shape[-2:], mode="bilinear", align_corners=False))
        p5 = self.output_convs[0](c5 + self.linear_proj[3](
            F.interpolate(bk_feature[3], size=c5.shape[-2:], mode="bilinear", align_corners=False)))

        # print("c2", c2.size())
        # print("c3", c3.size())
        # print("c4", c4.size())
        # print("c5", c5.size())
        # print("out4", out4.size())
        # print("out3", out3.size())
        # print("out2", out2.size())
        # exit()

        p4 = self.output_convs[1](out4)
        p3 = self.output_convs[2](out3)
        p2 = self.output_convs[3](out2)


        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(p2, p3, p4, p5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm = c2_ffm + c2
                c3_ffm = c3_ffm + c3
                c4_ffm = c4_ffm + c4
                c5_ffm = c5_ffm + c5
        p2 = c2_ffm
        # ____________________________
        multiscale_feature = [p2, p3, p4, p5]
        return multiscale_feature





class FPEM(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)



        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        # up阶段
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))


        # down 阶段
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))

        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x