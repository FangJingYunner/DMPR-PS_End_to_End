"""Defines the detector network structure."""
import torch
from torch import nn
from model.network import define_halve_unit, define_detector_block
from model.cspdarknet import CSPDarkNet
from model.stdcnet import STDC1Net
from thop import profile

class YetAnotherDarknet(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""
    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []
        # 0
        layers += [nn.Conv2d(input_channel_size, depth_factor, kernel_size=3,
                             stride=1, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(depth_factor)]
        layers += [nn.LeakyReLU(0.1)]
        # 1
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 2
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 3
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 4
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 5
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        self.model = nn.Sequential(*layers)

    def forward(self, *x):
        return self.model(x[0])


class DirectionalPointDetector(nn.modules.Module):
    """Detector for point with direction."""
    def __init__(self, input_channel_size, depth_factor, output_channel_size):
        super(DirectionalPointDetector, self).__init__()
        self.extract_feature = YetAnotherDarknet(input_channel_size,
                                                depth_factor)
        # self.extract_feature = CSPDarkNet()
        # self.extract_feature = STDC1Net(num_classes=1000, dropout=0.00, block_num=4)
        layers = []
        # layers += define_detector_block(16 * depth_factor)
        # layers += define_detector_block(16 * depth_factor)
        layers += [nn.Conv2d(32 * depth_factor, output_channel_size,
                             kernel_size=1, stride=1, padding=0, bias=False)]
        self.predict = nn.Sequential(*layers)

    def forward(self, *x):
        prediction = self.predict(self.extract_feature(x[0]))
        # prediction = self.predict(self.extract_feature(x[0])[0])
        
        
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        # point_pred, angle_pred = torch.split(prediction, 3, dim=1)
        point_pred = prediction[:,0:3,...]
        angle_pred = prediction[:, 3:, ...]
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        return torch.cat((point_pred, angle_pred), dim=1)
    
    
if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = DirectionalPointDetector(3, 16, 7)
    input_ = torch.randn((1, 3, 512, 512))
    output = model(input_)
    for i in output:
        print(i.shape)
        
    flops,params = profile(model, inputs=(input_,))
    # print(flops)
    print(params)
    print('  + Number of FLOPs: %.2fM' % (flops / 1e6 / 2))