import torch
from torch import nn
from model_slot_det.utils import *
from model_slot_det.pplcnet import light
from model_slot_det.pplcnet_res import lightres
from thop import profile
from model.stdcnet import STDC1Net
from model.cspdarknet import CSPDarkNet
# pplcnet 百度的
# arXiv: https://arxiv.org/pdf/2109.15099.pdf
# code:  https://github.com/PaddlePaddle/PaddleClas


class YOLOX(nn.Module):

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


class YOLOPAFPN(nn.Module):

    def __init__( self, in_dim,
                 depth=1.0, width=1.0, 
                 inc=[256, 512, 1024], 
                 expand=0.5,
                 depthwise=False,
                 features=("c3", "c4", "c5"),
                 model_type = "STDC1Net"
                 ):
        super().__init__()
        

        if model_type == 'light':
            self.backbone = light(width, False,
                                  in_chans=in_dim)
        elif model_type == 'lightres':
            self.backbone = lightres(width, False,
                                  in_chans=in_dim)
        elif model_type == 'STDC1Net':
            self.backbone = STDC1Net(base=int(64*width))
            
        self.features = features
        self.inc = inc
        ind = [int(dim*width) for dim in inc]
        dep = round(3 * depth)
        Conv = BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(ind[2], ind[1], 1, 1)
        self.C3_p4 = CSPLayer(ind[2], ind[1], dep, True, expand)
        
        self.reduce_conv1 = BaseConv(ind[1], ind[0], 1, 1)
        self.C3_p3 = CSPLayer(ind[1], ind[0], dep, True, 1)

        # bottom up
        self.bu_conv2 = Conv(ind[0], ind[0], 3, 2)
        self.C3_n3 = CSPLayer(ind[1], ind[1], dep, True, expand)

        self.bu_conv1 = Conv(ind[1], ind[1], 3, 2)
        self.C3_n4 = CSPLayer(ind[2], ind[2], dep, True, expand)

    def forward(self, x):

        out_chans = self.backbone(x)
        # [x2, x1, x0] = [out_chans[f] for f in self.features]
        (x2, x1, x0) = out_chans

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = pan_out0
        return outputs


class YOLOXHead(nn.Module):
    def __init__(self,
                 head_width=1.0, yolo_width=1.0, 
                 inc=[256, 512, 1024]):
        super().__init__()
        # self.config = config
        self.n_anchors = 1
        self.inc = inc
        Conv = BaseConv # don't use DWConv, 调整width depth
        ind = int(256 * head_width)
        oud = int(256 * head_width)
        input_c = int(self.inc[-1] * yolo_width)
        self.cls_convs = nn.Sequential(
            Conv(ind, oud, 3, 1), 
            Conv(ind, oud, 3, 1))
        self.reg_convs = nn.Sequential(
            Conv(ind, oud, 3, 1),
            Conv(ind, oud, 3, 1))
        self.entryline_preds = nn.Conv2d(ind, 2, 1)
        self.sepline_preds = nn.Conv2d(ind, 2, 1)
        self.reg_preds = nn.Conv2d(ind, 2, 1)
        self.obj_preds = nn.Conv2d(ind, 1, 1)
        self.stems = BaseConv(input_c, oud, 1, 1)

    def forward(self, x):

        x = self.stems(x)
        cls_x = x
        reg_x = x

        # decoupled head 解耦头
        cls_feat = self.cls_convs(cls_x)
        entryline_output = self.entryline_preds(cls_feat)  # 2 channel
        sepline_output = self.sepline_preds(cls_feat)  # 2 channel

        reg_feat = self.reg_convs(reg_x)
        position_output = self.reg_preds(reg_feat)  # 2 channel
        confidence_output = self.obj_preds(reg_feat)# 1 channel

        output = torch.cat([
            confidence_output.sigmoid(),# (0, 1)  confidence
            position_output.sigmoid(),  # (0, 1)  offset x, y
            sepline_output.tanh(),      # (-1, 1) sepline_x和y
            entryline_output.tanh(),    # (-1, 1) entryline_x和y
        ], 1)

        return output
    # 可考虑对conv和bn在这里做初始化


def get_model():
    # super-parameters in config
    depth = 1/3
    width = 0.5
    expand = 0.5
    ind = 3 # 1:gray 3:RGB, 3 is better, 1 is general
    inc = [256, 512, 1024]
    backbone = YOLOPAFPN(ind, depth, width, inc, expand, False,model_type = "STDC1Net")
    head = YOLOXHead( width, width,inc)
    model = YOLOX(backbone, head)

    return model

# use AdamW, lr衰减 batch_size=64
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)


if __name__ == "__main__":
    model = get_model()
    model.eval()
    x = torch.randn(1, 3, 416, 416)
    y = model(x)
    print(model)
    # torch.save(model.state_dict(), 'cat.pth')
    # for feature in y:
    # print(feature.shape)
    #     print(feature)
    flops,params = profile(model, inputs=(x,))
    # print(flops)
    print(params)
    print('  + Number of FLOPs: %.2fM' % (flops / 1e6 / 2))
    