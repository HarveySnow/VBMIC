import torch
from torch import nn
from models.vbmic import *
# from compressai.layers.quantization import quantize_ste

# 为VBMIC模型添加质量伸缩因子

class VBMIC_VAE(VBMIC):
    def __init__(self, N=128, M=192, decode_atten=JointContextTransfer):
        super().__init__(N, M, decode_atten)
        # 添加质量伸缩因此学习参数
        # 添加质量伸缩因子学习参数
        self.scaling_factors = nn.Parameter(torch.ones(1, M, 1, 1))
        self.bias_terms = nn.Parameter(torch.zeros(1, M, 1, 1))

    def forward(self, x):
        x_left, x_right = x[0], x[1]  # x.chunk(2, 1)
        # 编码
        y_left, y_right = self.encoder(x_left), self.encoder(x_right)
        # 编码后先进行SF操作 线性变换
        y_left = y_left * self.scaling_factors + self.bias_terms
        y_right = y_right * self.scaling_factors + self.bias_terms
        # 超先验部分 返回：最终输出 | 无损分布概率 | 熵解码结果
        left_params, z_left_likelihoods, z_left_hat = self.hyperprior(y_left, out_z=True)
        right_params, z_right_likelihoods, z_right_hat = self.hyperprior(y_right, out_z=True)
        # 进行高斯条件量化
        y_left_hat = self.gaussian_conditional.quantize(
            y_left, "noise" if self.training else "dequantize"
        )
        y_right_hat = self.gaussian_conditional.quantize(
            y_right, "noise" if self.training else "dequantize"
        )
        # 对高斯条件量化的输出获取mask后的上下文
        ctx_left_params = self.context_prediction(y_left_hat)
        ctx_right_params = self.context_prediction(y_right_hat)
        # 对上下文和超先验输出进行熵编码获得均值和方差
        gaussian_left_params = self.entropy_parameters(torch.cat([left_params, ctx_left_params], 1))
        gaussian_right_params = self.entropy_parameters(torch.cat([right_params, ctx_right_params], 1))
        # 分割出均值和方差
        left_means_hat, left_scales_hat = gaussian_left_params.chunk(2, 1)
        right_means_hat, right_scales_hat = gaussian_right_params.chunk(2, 1)
        # 利用均值方差进行高斯条件压缩
        _, y_left_likelihoods = self.gaussian_conditional(y_left, left_scales_hat, means=left_means_hat)
        _, y_right_likelihoods = self.gaussian_conditional(y_right, right_scales_hat, means=right_means_hat)

        # 利用均值进行随机化量化
        y_left_ste, y_right_ste = quantize_ste(y_left - left_means_hat) + left_means_hat, quantize_ste(
            y_right - right_means_hat) + right_means_hat
        # 量化后进行ISF操作 逆缩放
        y_left_ste = (y_left_ste - self.bias_terms) / self.scaling_factors
        y_right_ste = (y_right_ste - self.bias_terms) / self.scaling_factors
        # 经过第一个JCT模块
        y_left, y_right = self.atten_3(y_left_ste, y_right_ste)
        # 经过解码器前半部分后在经过第二个JCT模块
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        # 经过解码器后半部分完成解码重建
        x_left_hat, x_right_hat = self.decoder_2(y_left), self.decoder_2(y_right)
        '''
        - x_hat:重建图像;
        - likelihoods: 压缩过程中的概率,包含y和z的编码似然概率,用于无损解压。
        - feature: 一些中间特征,用于其他分析。具体包含:
            y_left/right_ste: 随机量化后的编码特征
            z_left/right_hat: z熵编码后的结果
            left/right_means_hat: 高斯均值参数
        '''
        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [{"y": y_left_likelihoods, "z": z_left_likelihoods},
                            {"y": y_right_likelihoods, "z": z_right_likelihoods}],
            "feature": [y_left_ste, y_right_ste, z_left_hat, z_right_hat, left_means_hat, right_means_hat],
        }

    # 编码包括编码器编码和超先验编码过程
    def encode(self, x):
        y = self.encoder(x)

        # SF
        y = y * self.scaling_factors + self.bias_terms

        params, z_hat, z_strings = self.hyperprior.compress(y)
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z_hat.size()[-2:]}

    # 解码
    def decompress(self, left_dict, right_dict):
        y_left_hat = self.decode(left_dict["strings"], left_dict["shape"])
        y_right_hat = self.decode(right_dict["strings"], right_dict["shape"])

        # ISF
        y_left_hat = (y_left_hat - self.bias_terms) / self.scaling_factors
        y_right_hat = (y_right_hat - self.bias_terms) / self.scaling_factors

        y_left, y_right = self.atten_3(y_left_hat, y_right_hat)
        y_left, y_right = self.atten_4(self.decoder_1(y_left), self.decoder_1(y_right))
        x_left_hat, x_right_hat = self.decoder_2(y_left).clamp_(0, 1), self.decoder_2(y_right).clamp_(0, 1)
        return {
            "x_hat": [x_left_hat, x_right_hat],
        }

    # 固定模型参数（除质量收缩因子以外）,即在训练时关闭编码器参数梯度和更新。
    def fix_model(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False
        for p in self.atten_3.parameters():
            p.requires_grad = False
        for p in self.decoder_1.parameters():
            p.requires_grad = False
        for p in self.atten_4.parameters():
            p.requires_grad = False
        for p in self.decoder_2.parameters():
            p.requires_grad = False

    def fix_model_else_encoder(self):
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False
        for p in self.atten_3.parameters():
            p.requires_grad = False
        for p in self.atten_4.parameters():
            p.requires_grad = False