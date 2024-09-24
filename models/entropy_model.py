import torch
import math
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import quantize_ste
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
import torch.nn.functional as F
import copy

class CheckMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask: A
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == "A":
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        else:
            self.mask[:, :, 0::2, 0::2] = 1
            self.mask[:, :, 1::2, 1::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        return out

# 超先验网络
class Hyperprior(CompressionModel):
    # 输入通道 | 瓶颈层通道 | 输出通道
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192):
        # self.entropy_bottleneck是Hyperprior类继承自CompressionModel的一个属性。EntropyBottleneck实现了基于全概率的压缩机制。
        # 算术编码应该包含在entropy_bottleneck中
        super().__init__(entropy_bottleneck_channels=mid_planes)
        # 超先验编码器
        self.hyper_encoder = nn.Sequential(
            conv(in_planes, mid_planes, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
        )
        if out_planes == 2 * in_planes:
            # 超先验解码器
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, in_planes * 3 // 2, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(in_planes * 3 // 2, out_planes, stride=1, kernel_size=3),
            )
        else:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(mid_planes, out_planes, stride=1, kernel_size=3),
            )

    def forward(self, y, out_z=False):
        # 编码
        z = self.hyper_encoder(y)
        # 压缩后的变量 _: 这是经过压缩后的z的值,已经丢失了部分信息；压缩likelihoods: 这是z各元素的概率分布描述,可以用于无损解压。
        _, z_likelihoods = self.entropy_bottleneck(z)
        # 获取压缩过程中用到的均值偏移量z_offset。
        z_offset = self.entropy_bottleneck._get_medians()
        '''
        z_offset是z每个元素对应的均值,反映了z的整体分布情况。
        加入z_offset的压缩过程可以理解为:用z_offset消除z中的冗余信息,得到去均值的残差信号。仅压缩去均值后的残差信号。解压时再加回z_offset还原原始z。
        这种压缩思路的优点是:去均值可以更有效地减少冗余,提高压缩率。只压缩残差信号,量化误差也更小。加回均值可以无损还原。
        '''
        #  对z减去偏移量,进行整数化rounding,得到量化后的z_hat。将量化后的z_hat加回偏移量,还原。
        # quantize_ste表示Stochastic rounding,也就是随机化量化。一定的概率向上取整一定的概率向下取整
        z_hat = quantize_ste(z - z_offset) + z_offset
        # 超先验解码输出
        params = self.hyper_decoder(z_hat)
        # 是否返回比特流
        if out_z:
            return params, z_likelihoods, z_hat
        else:
            return params, z_likelihoods
    # 包含整个压缩过程的编码、量化、反量化、解码 返回从本输入获得的辅助信息z^hat和解码结果、中间比特流
    def compress(self, y):
        z = self.hyper_encoder(y)
        # compress方法会根据z的分布,学习它的概率模型,并基于此模型使用算术编解码的方式无损压缩z，得到压缩后的字符串z_strings。
        z_strings = self.entropy_bottleneck.compress(z)
        # 可以无损的恢复z,即重建结果z_hat应与原始z相同。
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.hyper_decoder(z_hat)
        return params, z_hat, z_strings #{"strings": z_string, "shape": z.size()[-2:]}
    # 量化结果解压缩 返回：从本输入获得的辅助信息z^hat和解码结果
    def decompress(self, strings, shape):
        #assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        params = self.hyper_decoder(z_hat)
        return params, z_hat
