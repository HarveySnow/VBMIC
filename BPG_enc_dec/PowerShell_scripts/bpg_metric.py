import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_msssim import ms_ssim
import numpy as np
import os
from PIL import Image
from torchvision import transforms

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

# 递归获取目录及其子目录下的所有文件
def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

# 将图像文件转换为 Tensor
def image_to_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor

bpg_enc_dir = "./compress/bpg/enc"
bpg_dec_dir = "./compress/bpg/dec"

all_enc_files = get_all_files(bpg_enc_dir)
all_dec_files = get_all_files(bpg_dec_dir)

for enc,dec in zip(all_enc_files, all_dec_files):
    print(enc,dec)
    org_frame = image_to_tensor(enc)
    rec_frame = image_to_tensor(dec)
    psnr_float, ms_ssim_float = compute_metrics_for_frame(org_frame, rec_frame)
    print("PSNR:", psnr_float.item()) 
    print("MS-SSIM:", ms_ssim_float.item())
