import argparse
import json
import math
import sys
import os
import time
import struct

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
import compressai

from compressai.zoo.pretrained import load_pretrained
from models.vbmic import *
from models.vbmic_vae import VBMIC_VAE
from models.entropy_model import *
from torch.hub import load_state_dict_from_url
from lib.utils import CropCityscapesArtefacts, MinimalCrop

from compressai import ans
import re

import matplotlib.pyplot as plt
import torchvision.utils as vutils
# from compressai.zoo import bmshj2018_factorized

# 通用的图像编码模型评估脚本,可以灵活配置不同的数据集、模型、参数,进行编码效率和重建质量的评测。

# 这段代码实现了根据不同的数据集,从指定的根目录中收集左右图像序列的功能。数据集名称data_name\根目录rootpath
def collect_images(data_name:str, rootpath: str):
    # 如果是cityscapes,则从leftImg8bit/test下面递归搜索所有.png文件,对于每个左图像,构造对应的右图像路径。
    if data_name == 'cityscapes':
        left_image_list, right_image_list = [], []
        path = Path(rootpath)
        for left_image_path in path.glob(f'leftImg8bit/test/*/*.png'):
            left_image_list.append(str(left_image_path))
            right_image_list.append(str(left_image_path).replace("leftImg8bit", 'rightImg8bit'))
    # 如果是instereo2k,直接从test目录下的子文件夹中取出左右图像路径。
    elif data_name == 'instereo2k':
        path = Path(rootpath)
        path = path / "test"   
        folders = [f for f in path.iterdir() if f.is_dir()]
        left_image_list = [f / 'left.png' for f in folders]
        right_image_list = [f / 'right.png' for f in folders] #[1, 3, 860, 1080], [1, 3, 896, 1152]
    # 如果是wildtrack,则从C1和C4目录分别读取图像,C1为左,C4为右。
    elif data_name == 'wildtrack':
        C1_image_list, C4_image_list = [], []
        path = Path(rootpath)
        for image_path in path.glob(f'images/C1/*.png'):
            if int(image_path.stem) > 2000:
                C1_image_list.append(str(image_path))
                C4_image_list.append(str(image_path).replace("C1", 'C4'))
        left_image_list, right_image_list = C1_image_list, C4_image_list

    return [left_image_list, right_image_list]

# 评估一个模型在数据集上的整体平均表现
# 接收所有结果文件路径filepaths作为参数
def aggregate_results(filepaths: List[Path]) -> Dict[str, Any]:
    # 初始化一个字典metrics来保存每个指标的结果
    metrics = defaultdict(list)
    # 遍历所有结果文件:
    # sum
    for f in filepaths:
        with f.open("r") as fd:
            data = json.load(fd)
        for k, v in data["results"].items():
            metrics[k].append(v)

    # normalize
    # 最后遍历metrics,计算每个指标结果的平均值(np.mean)。
    agg = {k: np.mean(v) for k, v in metrics.items()}
    return agg


# 实现了对tensor图像进行zero padding的功能,使其高度和宽度都能够被指定数p整除。
def pad(x: Tensor, p: int = 2 ** (4 + 3)) -> Tuple[Tensor, Tuple[int, ...]]:
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x = F.pad(x, padding, mode="constant", value=0)
    return x, padding

# 对已padding的图像去除padding,还原图像范围
def crop(x: Tensor, padding: Tuple[int, ...]) -> Tensor:
    return F.pad(x, tuple(-p for p in padding))

# 计算两张图像之间的PSNR和MS-SSIM指标的功能。
def compute_metrics_for_frame(
    org_frame: Tensor,
    rec_frame: Tensor,
    device: str = "cpu",
    max_val: int = 255,):
    
    #psnr_float = -10 * torch.log10(F.mse_loss(org_frame, rec_frame))
    #ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=1.0)

    org_frame = (org_frame * max_val).clamp(0, max_val).round()
    rec_frame = (rec_frame * max_val).clamp(0, max_val).round()
    mse_rgb = (org_frame - rec_frame).pow(2).mean()
    psnr_float = 20 * np.log10(max_val) - 10 * torch.log10(mse_rgb)
    ms_ssim_float = ms_ssim(org_frame, rec_frame, data_range=max_val)
    return psnr_float, ms_ssim_float

# 这段代码实现了计算比特率(bpp)的功能。
# likelihoods: 是压缩过程中生成的各部分的likelihood概率。一般是从压缩模型的输出中取得。是一个字典。
# num_pixels: 是图像中的总像素数
def compute_bpp(likelihoods, num_pixels):
    bpp = sum(
        (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
        for likelihood in likelihoods.values()
    )
    return bpp

# 读取图像文件并进行预处理,返回一个Tensor格式的图像。
def read_image(crop_transform, filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    # 如果传入了crop_transform,则调用它对图像进行裁剪/变换。
    if crop_transform is not None:
        img = crop_transform(img)
    return transforms.ToTensor()(img)

def save_encoded_features_as_images(features, output_dir, file_prefix):
    """
    将每个特征图的每个通道保存为单独的图像
    :param features: 特征图张量，形状为 [1, channels, height, width]
    :param output_dir: 保存图像的目录路径
    :param file_prefix: 保存的文件前缀，用于区分不同的输入图像
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 从特征图张量中获取通道数
    channels = features.size(1)

    # 遍历所有通道
    for channel in range(channels):
        # 提取单个通道的特征图并进行归一化
        channel_features = features[0, channel, :, :].unsqueeze(0).unsqueeze(0)
        channel_min = channel_features.min()
        channel_max = channel_features.max()

        # 避免除0错误，在这里添加一个小的epsilon值
        epsilon = 1e-5
        norm_channel_features = (channel_features - channel_min) / (channel_max - channel_min + epsilon)

        # 使用热图颜色映射
        cmap = plt.get_cmap('jet')
        norm_channel_features_np = norm_channel_features.squeeze().cpu().detach().numpy()
        colored_features = cmap(norm_channel_features_np)[:, :, :3]  # 取前三个通道为RGB
        colored_features = torch.from_numpy(colored_features).permute(2, 0, 1).float()  # 转换为torch张量

        # 保存当前通道的特征图
        feature_image_path = output_dir / f"{file_prefix}_channel_{channel}.png"
        vutils.save_image(colored_features, feature_image_path)

    print("Sucess.")

# 可视化操作
def visualize_features(IFrameCompressor: nn.Module, filepaths: List[Path], output_dir: Path,
                       crop_transform=None) -> None:
    device = next(IFrameCompressor.parameters()).device
    num_frames = len(filepaths)

    # Convert output_dir to a Path object if it's not already
    output_dir = Path(output_dir)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个图像
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            # 读取图像预处理为tensor
            x = read_image(crop_transform, filepaths[i]).unsqueeze(0).to(device)
            print(f"Processing image: {filepaths[i]}, shape: {x.size()}")
            features = IFrameCompressor.encoder(x)

            # 保存特征图为图像
            # feature_image_path = output_dir / f"VBMIC_feature_maps_{i}"
            save_encoded_features_as_images(features, output_dir, f"VBMIC_feature_maps_{i}")

            #pbar.update(1)
            break

# 对一个图像序列使用给定的编码模型进行编码、解码和评估
@torch.no_grad()
def eval_model(IFrameCompressor:nn.Module, left_filepaths: Path, right_filepaths: Path, **args: Any) -> Dict[str, Any]:
    print("eval_model begin...")
    # 获取模型设备device和图像序列长度num_frames。
    device = next(IFrameCompressor.parameters()).device
    num_frames = len(left_filepaths)
    # 像素最大值255
    max_val = 2**8 - 1
    # 定义结果字典results来记录各指标。
    results = defaultdict(list)
    if args["crop"]:
        crop_transform = CropCityscapesArtefacts() if args["data_name"] == "cityscapes" else MinimalCrop(min_div=64)
    else:
        crop_transform = None

    results = defaultdict(list)
    # 遍历每个图像
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            # 读取图像预处理为tensor
            x_left = read_image(crop_transform, left_filepaths[i]).unsqueeze(0).to(device)
            num_pixels = x_left.size(2) * x_left.size(3)
            x_right = read_image(crop_transform, right_filepaths[i]).unsqueeze(0).to(device)
            # left图像压缩时间
            start = time.time()
            out_enc_left = IFrameCompressor.encode(x_left)
            enc_left_time = time.time() - start

            # 可视化特征图
            print("make the picture of features……")
            features_left = IFrameCompressor.encoder(x_left)
            #print(features_left)
            save_encoded_features_as_images(features_left, "output/features_img/VBMIC")

            # right图像压缩时间
            start = time.time()
            out_enc_right = IFrameCompressor.encode(x_right)
            enc_right_time = time.time() - start
            # 解码时间
            start = time.time()
            out_dec = IFrameCompressor.decompress(out_enc_left, out_enc_right)
            dec_time = time.time() - start
            # 得到重建结果
            x_left_rec, x_right_rec = out_dec["x_hat"][0], out_dec["x_hat"][1]
            # 评估指标
            metrics = {}
            metrics["left-psnr-float"], metrics["left-ms-ssim-float"] = compute_metrics_for_frame(
                x_left, x_left_rec, device, max_val)
            metrics["right-psnr-float"], metrics["right-ms-ssim-float"] = compute_metrics_for_frame(
                x_right, x_right_rec, device, max_val)
            
            metrics["psnr-float"] = (metrics["left-psnr-float"]+metrics["right-psnr-float"])/2
            metrics["ms-ssim-float"] = (metrics["left-ms-ssim-float"]+metrics["right-ms-ssim-float"])/2

            metrics["left_bpp"] = torch.tensor(sum(len(s[0]) for s in out_enc_left["strings"]) * 8.0 / num_pixels)
            metrics["right_bpp"] = torch.tensor(sum(len(s[0]) for s in out_enc_right["strings"]) * 8.0 / num_pixels)
            metrics["bpp"] = (metrics["left_bpp"] + metrics["right_bpp"])/2
            # 简化代码逻辑,只用处理tensor类型。不用区分不同变量的格式。
            enc_left_time = torch.tensor(enc_left_time)
            enc_right_time = torch.tensor(enc_right_time)
            dec_time = torch.tensor(dec_time)
            metrics["enc_left_time"] = enc_left_time
            metrics["enc_right_time"] = enc_right_time
            metrics["enc_time"] = max(enc_left_time, enc_right_time) #torch.max(torch.cat([enc_left_time, enc_right_time], dim=0))
            metrics["enc_average_time"] = (enc_left_time + enc_right_time)/2
            
            metrics["dec_time"] = dec_time
            metrics["dec_average_time"] = dec_time/2

            #print(metrics)
            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)
    # 汇总每个指标平均值，返回结果字典seq_results
    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results

# 熵估计模型评估
@torch.no_grad()
def eval_model_entropy_estimation(IFrameCompressor:nn.Module, left_filepaths: Path, right_filepaths: Path, **args: Any) -> Dict[str, Any]:
    device = next(IFrameCompressor.parameters()).device
    num_frames = len(left_filepaths) 
    max_val = 2**8 - 1
    results = defaultdict(list)
    if args["crop"]:
        crop_transform = CropCityscapesArtefacts() if args["data_name"] == "cityscapes" else MinimalCrop(min_div=64)
    else:
        crop_transform = None

    with tqdm(total=num_frames) as pbar: #97: 0-96
        for i in range(num_frames):

            x_left = read_image(crop_transform, left_filepaths[i]).unsqueeze(0).to(device)
            num_pixels = x_left.size(2) * x_left.size(3)

            x_right = read_image(crop_transform, right_filepaths[i]).unsqueeze(0).to(device)
            left_height, left_width = x_left.shape[2:]
            right_height, right_width = x_right.shape[2:]

            out = IFrameCompressor([x_left, x_right])
            x_left_rec, x_right_rec = out["x_hat"][0], out["x_hat"][1]
            left_likelihoods, right_likelihoods = out["likelihoods"][0], out["likelihoods"][1]

            # 把解码后的图像限制在[0,1]的范围内。限制tensor中元素的范围,防止越界。
            # 由于解码误差的影响,输出的x_left_rec和x_right_rec中的像素值可能会小于0或者大于1。
            x_left_rec = x_left_rec.clamp(0, 1)
            x_right_rec = x_right_rec.clamp(0, 1)
            # 评估指标
            metrics = {}
            metrics["left-psnr-float"], metrics["left-ms-ssim-float"] = compute_metrics_for_frame(
                x_left, x_left_rec, device, max_val)
            metrics["right-psnr-float"], metrics["right-ms-ssim-float"] = compute_metrics_for_frame(
                x_right, x_right_rec, device, max_val)
            
            metrics["psnr-float"] = (metrics["left-psnr-float"]+metrics["right-psnr-float"])/2
            metrics["ms-ssim-float"] = (metrics["left-ms-ssim-float"]+metrics["right-ms-ssim-float"])/2

            metrics["left_bpp"] = compute_bpp(left_likelihoods, num_pixels)
            metrics["right_bpp"] = compute_bpp(right_likelihoods, num_pixels)
            metrics["bpp"] = (metrics["left_bpp"] + metrics["right_bpp"])/2

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results

# 测试模型的核心逻辑
def run_inference(
    filepaths,
    IFrameCompressor: nn.Module, 
    outputdir: Path,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any):
    # 从文件路径中解析出左右两张图像。
    left_filepath, right_filepath = filepaths[0], filepaths[1]
    #sequence_metrics_path = Path(outputdir) / f"{trained_net}.json"

    #if force:
    #    sequence_metrics_path.unlink(missing_ok=True)
    # PyTorch中的一个自动混合精度(Auto Mixed Precision, AMP)的上下文管理器。
    # 使用半精度浮点数(FP16)进行部分计算,可以加速深度学习模型的训练及推理过程,同时保持与FP32精度相近的结果。
    with amp.autocast(enabled=args["half"]):
        with torch.no_grad():
            if entropy_estimation:
                metrics = eval_model_entropy_estimation(IFrameCompressor, left_filepath, right_filepath, **args)
            else:
                metrics = eval_model(IFrameCompressor, left_filepath, right_filepath, **args)
    return metrics

# 解析命令行参数
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stereo image compression network evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 测试图像目录
    parser.add_argument("-d", "--dataset", type=str, default='data/cityscapes/', help="sequences directory")
    # 测试数据集名称
    parser.add_argument("--data-name", type=str, default='cityscapes', help="sequences directory")
    # 输出目录
    parser.add_argument("--output", type=str, default='output', help="output directory")
    # lambda可选参数用于避免maskdecay操作报错
    parser.add_argument("--lambda", dest="lmbda",
        type=float,
        default=4096,
        help="Bit-rate distortion parameter (default: %(default)s)",)
    # 模型架构
    parser.add_argument(
        "-im",
        "--IFrameModel",
        default="VBMIC",
        help="Model architecture (default: %(default)s)",
    )
    # 模型质量参数
    parser.add_argument("-iq", "--IFrame_quality", type=int, default=4, help='Model quality')
    # 模型路径 VBMIC_VAE/lamda4096/train-run3
    parser.add_argument("--i_model_path", type=str, default='checkpoints/cityscapes/mse/VBMIC/lamda64/train-run2/ckpt.pth.tar',help="Path to a checkpoint")
    # 是否裁剪图像
    parser.add_argument("--crop", action="store_true", help="use crop")
    # 是否使用GPU
    parser.add_argument("--cuda", action="store_true",default=True, help="use cuda")
    # 是否使用混合精度
    parser.add_argument("--half", action="store_true", help="use AMP")
    # 是否使用熵估计
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",default=True,
        help="use evaluated entropy estimation (no entropy coding)",
    )
    # 熵编码器选择
    parser.add_argument(
        "-c",
        "--entropy-coder",
        default="ans",
        help="entropy coder (default: %(default)s)",
    )
    # 是否保留比特率文件
    parser.add_argument(
        "--keep_binaries",
        action="store_true",
        help="keep bitstream files in output directory",
    )
    # 详细模式
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    # 评估指标
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    # cpu数量
    parser.add_argument("--cpu_num", type=int, default=4)
    return parser


def main(args: Any = None) -> None:
    # 解析参数
    # sys.argv[1:]表示命令行参数列表,它包含运行Python程序时传入的参数。
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)
    # 这里根据entropy_estimation来决定description的取值:
    # 如果是熵估计模式,则description为entropy-estimation
    # 否则为所选的entropy_coder名称
    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )
    filepaths = collect_images(args.data_name, args.dataset)

    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    # 如果是cpu，设置多线程并行
    if device == "cpu":
        cpu_num = args.cpu_num # 这里设置成你想运行的CPU个数
        # 环境变量控制……使用的线程数
        os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)
 
    # LDMIC是标准的全卷积网络结构,使用类似跳连接的上下文传递模块。
    # LDMIC_checkboard在编解码器中间加入了棋盘式遮挡模块,可以强制网络学习到更丰富的上下文表示。
    if args.IFrameModel == "VBMIC":
        IFrameCompressor = VBMIC(N=64, M=64, decode_atten=JointContextTransfer)
    elif args.IFrameModel == "VBMIC_VAE":
        IFrameCompressor = VBMIC_VAE(N=96, M=192, decode_atten=JointContextTransfer)
    elif args.IFrameModel == "VBMIC_checkboard":
        IFrameCompressor = VBMIC_checkboard(N=192, M=192, decode_atten=JointContextTransfer)
                      
    IFrameCompressor = IFrameCompressor.to(device)
    if args.i_model_path:
        print("Loading model:", args.i_model_path)
        # 加载模型检查点文件,并将参数和网络结构加载到内存中 map_location参数用于将参数映射到指定设备device(CPU/GPU)。
        checkpoint = torch.load(args.i_model_path, map_location=device)
        # 从检查点中取出状态字典state_dict,它保存了模型的参数。
        IFrameCompressor.load_state_dict(checkpoint["state_dict"])

        # # 如果是vae模型，需要从指定的位置加载缩放因子参数
        # saved_params_path = "checkpoints/cityscapes/mse/LDMIC_VAE/128_ckpt.pth.tar"
        # saved_state_dict = torch.load(saved_params_path)
        # # 从 saved_state_dict 中提取 scaling_factors 和 bias_terms 参数
        # scaling_factors = saved_state_dict['scaling_factors']
        # bias_terms = saved_state_dict['bias_terms']
        # # 将参数加载到模型中
        # IFrameCompressor.scaling_factors.data.copy_(scaling_factors)
        # IFrameCompressor.bias_terms.data.copy_(bias_terms)
        # print("Load scaling_factors……")

        # 调用update方法确保模型参数都进行了同步。
        IFrameCompressor.update(force=True)
        # 设置模型为评估模式eval()
        IFrameCompressor.eval()


    # create output directory
    outputdir = args.output
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    # 定义默认的工厂函数来生成不存在键对应的默认值,避免key错误
    results = defaultdict(list)
    args_dict = vars(args)
    # 构造模型评估结果的名称
    trained_net = f"{args.IFrameModel}-{args.metric}-{description}"
    # 调用测试函数获取测试结果

    # 进行特征图可视化操作
    #visualize_features(IFrameCompressor, filepaths[0], outputdir)

    # 原来的eval操作
    metrics = run_inference(filepaths, IFrameCompressor, outputdir, trained_net=trained_net, description=description, **args_dict)

    # 将结果写入结果文件
    for k, v in metrics.items():
        results[k].append(v)

    output = {
        "name": f"{args.IFrameModel}-{args.metric}",
        "description": f"Inference ({description})",
        "results": results,
    }

    with (Path(f"{outputdir}/{args.IFrameModel}-{args.metric}-{description}.json")).open("wb") as f:
        f.write(json.dumps(output, indent=2).encode())
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
