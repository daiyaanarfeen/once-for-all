import torch
import numpy as np
import ofa
from ofa.model_zoo import ofa_net
from ofa.tutorial.evolution_finder import ArchManager
from ofa.utils import make_divisible

model = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
stages = [model.blocks[1+4*i: 1+4*(i+1)] for i in range(5)]
sampler = ArchManager()
sample = sampler.random_sample()

#for i, stage in enumerate(stages):
#    depth = sample['d'][i]
#    kernels = sample['ks'][4*i: 4*(i+1)]
#    expand_ratios = sample['e'][4*i: 4*(i+1)]

out_channels = [model.blocks[i].mobile_inverted_conv.point_linear.conv.conv.out_channels for i in range(1, 21)]
out_channels = [model.blocks[0].mobile_inverted_conv.point_linear.conv.out_channels] + out_channels
l2_squared = np.zeros([20, 3, 3], dtype=np.float)
for i in range(20):
    for k in [3, 5, 7]:
        for e in [3, 4, 6]:
            l2 = 0.0
            module = model.blocks[i+1].mobile_inverted_conv
            in_channel = out_channels[i]
            mid_channels = make_divisible(round(in_channel * e), 8)

            if module.inverted_bottleneck is not None:
                l2 += torch.norm(module.inverted_bottleneck.conv.conv.weight[:mid_channels, :in_channel, :, :]) ** 2
                l2 += torch.norm(module.inverted_bottleneck.bn.bn.weight[:mid_channels]) ** 2
                if module.inverted_bottleneck.bn.bn.bias is not None:
                    l2 += torch.norm(module.inverted_bottleneck.bn.bn.bias[:mid_channels]) ** 2

            l2 += torch.norm(module.depth_conv.conv.get_active_filter(mid_channels, k))**2
            l2 += torch.norm(module.depth_conv.bn.bn.weight[:mid_channels]) ** 2
            if module.depth_conv.bn.bn.bias is not None:
                l2 += torch.norm(module.depth_conv.bn.bn.bias[:mid_channels]) ** 2
            if hasattr(module.depth_conv, 'se'):
                se_channel = make_divisible(mid_channels // module.depth_conv.se.reduction, divisor=8)
                l2 += torch.norm(module.depth_conv.se.fc.reduce.weight[:se_channel, :mid_channels, :, :]) ** 2
                if module.depth_conv.se.fc.reduce.bias is not None:
                    l2 += torch.norm(module.depth_conv.se.fc.reduce.bias[:se_channel]) ** 2
                l2 += torch.norm(module.depth_conv.se.fc.expand.weight[:mid_channels, :se_channel, :, :]) ** 2
                if module.depth_conv.se.fc.expand.bias is not None:
                    l2 += torch.norm(module.depth_conv.se.fc.expand.bias[:mid_channels]) ** 2

            l2 += torch.norm(module.point_linear.conv.conv.weight[:out_channels[i+1], :mid_channels, :, :]) ** 2
            l2 += torch.norm(module.point_linear.bn.bn.weight[:out_channels[i+1]]) ** 2
            if module.point_linear.bn.bn.bias is not None:
                l2 += torch.norm(module.point_linear.bn.bn.bias[:out_channels[i+1]]) ** 2


            l2_squared[i, [3, 5, 7].index(k), [3, 4, 6].index(e)] = l2
