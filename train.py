import argparse
import math
import random
import sys
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from models.vbmic import *
from models.vbmic_vae import VBMIC_VAE
from lib.utils import get_output_folder, AverageMeter, save_checkpoint, StereoImageDataset
import numpy as np

import yaml
import wandb
import os
from tqdm import tqdm
from pytorch_msssim import ms_ssim

os.environ["WANDB_API_KEY"] = "123456abcdefg" # wandb api key

# 计算辅助损失
'''
输入是一个aux_loss组成的列表aux_list。
初始化aux_loss_sum为0。
遍历aux_list中的每个aux_loss,累加到aux_loss_sum。
如果backward为True,则对每个aux_loss调用.backward()方法进行反向传播,计算梯度。
最终返回所有aux_loss的累加和aux_loss_sum。
'''
def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    return aux_loss_sum

# 配置优化器
'''
从模型参数中分离主要参数和辅助参数:
主要参数是那些名字不包含“quantiles”的可训练参数。
辅助参数是名字包含“quantiles”的可训练参数。
检查两个参数集合不包含重复参数。
为主要参数创建Adam优化器optimizer,学习率为args.learning_rate。
为辅助参数创建Adam优化器aux_optimizer,学习率也是args.learning_rate。
返回这两个优化器optimizer和aux_optimizer。
这样在训练时,模型的参数就会分成两个组:
主要参数由optimizer进行优化
辅助参数(例如量化聚类中心)由aux_optimizer进行优化
'''
def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.learning_rate,
    )
    return optimizer, aux_optimizer

# 这段代码实现了模型在训练数据集上的单个epoch的训练过程。
def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, args):
    model.train()
    device = next(model.parameters()).device

    # 定义训练时使用的评估指标:如果是MSE,则指标为PSNR,如果是MS-SSIM,则指标为MS-SSIM dB
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')

    # 训练每个batch
    train_dataloader = tqdm(train_dataloader)
    print('Train epoch:', epoch)
    for i, batch in enumerate(train_dataloader):
        d = [frame.to(device) for frame in batch]
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
        #aux_optimizer.zero_grad()
        
        out_net = model(d)
        out_criterion = criterion(out_net, d, args.lmbda)

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if aux_optimizer is not None:
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
            aux_optimizer.step()
        else:
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)
        #out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        #aux_optimizer.step()

        loss.update(out_criterion["loss"].item())
        bpp_loss.update((out_criterion["bpp_loss"]).item())
        aux_loss.update(out_aux_loss.item())
        metric_loss.update(out_criterion[metric_name].item())
        
        left_bpp.update(out_criterion["bpp0"].item())
        right_bpp.update(out_criterion["bpp1"].item())

        if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
            left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
            right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
            left_db.update(left_metric)
            right_db.update(right_metric)
            metric_dB.update((left_metric+right_metric)/2)

        train_dataloader.set_description('[{}/{}]'.format(i, len(train_dataloader)))
        train_dataloader.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
            metric_dB_name:metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}
    # 返回一个字典,包含平均的loss,bpp,指标等信息。
    return out

# 模型在验证集上的测试过程
def test_epoch(epoch, val_dataloader, model, criterion, args):
    # model.eval()和 model.train() 是用于指定模型在训练模式或是评估(测试)模式下运行的方法。
    # pytorch会关闭dropout和batch normalization层的某些行为,从而使得模型在测试时更加稳定可靠。
    model.eval()
    device = next(model.parameters()).device

    # 评估指标
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')
    loop = tqdm(val_dataloader)

    # 遍历验证集 关闭梯度计算 加速测试
    with torch.no_grad():
        for i, batch in enumerate(loop):
            d = [frame.to(device) for frame in batch]
            
            out_net = model(d)
            out_criterion = criterion(out_net, d, args.lmbda)

            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)

            loss.update(out_criterion["loss"].item())
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            aux_loss.update(out_aux_loss.item())
            metric_loss.update(out_criterion[metric_name].item())
        
            left_bpp.update(out_criterion["bpp0"].item())
            right_bpp.update(out_criterion["bpp1"].item())

            if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
                left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
                right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
                left_db.update(left_metric)
                right_db.update(right_metric)
                metric_dB.update((left_metric+right_metric)/2)

            loop.set_description('[{}/{}]'.format(i, len(val_dataloader)))
            loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}

    # 返回一个字典,包含平均的loss,bpp,各项指标等。
    return out

# 命令行参数的解析
def parse_args(argv):
    # 命令行参数解析模块。其中 description 参数用于指定解析器的描述信息,会显示在帮助信息里。
    # 创建argparse.ArgumentParser对象parser来添加和解析命令行参数。
    parser = argparse.ArgumentParser(description="Example training script.")
    # 数据集路径
    parser.add_argument(
        "-d", "--dataset", type=str, default='data/cityscapes/', help="Training dataset"
    )
    # 数据集名称
    parser.add_argument(
        "--data-name", type=str, default='cityscapes', help="Training dataset"
    )
    # 模型名称
    parser.add_argument(
        "--model-name", type=str, default='VBMIC', help="Training dataset"
    )
    # dataloader的线程数
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Dataloaders threads (default: %(default)s)",
    )
    # 率失真参数
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=64,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    # batch大小
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: %(default)s)"
    )
    # 训练轮数
    parser.add_argument(
        "--epochs", type=int, default=400, help="number of training epochs (default: %(default)s)"
    )
    # 测试batch大小
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    # 裁剪patch大小
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    # 使用GPU
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    # 保存模型
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    # 使用resize或随机裁剪
    parser.add_argument(
        "--resize", action="store_true", default=True, help="training use resize or randomcrop"
    )
    # 随机种子
    parser.add_argument(
        "--seed", type=float, default=1, help="Set random seed for reproducibility"
    )
    # 梯度裁剪的值
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    # 预训练模型路径
    parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    # 使用的指标 mse或ms-ssim
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms-ssim")
    # 学习率
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: %(default)s)",
    )
    # 通过parser.parse_args解析输入的参数,并返回结果args。
    # 这样在训练脚本中就可以方便地使用args访问到解析后的参数,比如args.epochs
    args = parser.parse_args(argv)
    return args

# 训练主逻辑
def main(argv):
    # 调用前面定义的函数解析命令行参数
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    # 保存这些参数用于复现实验,这里将其序列化为YAML格式，default_flow_style=False表示YAML字符串采用多行格式,更易读。
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Warning, the order of the transform composition should be kept.
    # 构建训练集和验证集的数据加载器。
    train_dataset = StereoImageDataset(ds_type='train', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size, resize=args.resize)
    test_dataset = StereoImageDataset(ds_type='test', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size, resize=args.resize)


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    if args.model_name == "VBMIC":
        # net = LDMIC(N=192, M=192, decode_atten=JointContextTransfer)
        net = VBMIC(N=64, M=64, decode_atten=JointContextTransfer)
    elif args.model_name == "VBMIC_VAE":
        net = VBMIC_VAE(N=96, M=192, decode_atten=JointContextTransfer)
    elif args.model_name == "VBMIC_checkboard":
        net = VBMIC_checkboard(N=192, M=192, decode_atten=JointContextTransfer)

    net = net.to(device) 

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.5) #optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)
    if args.metric == "mse":
        criterion = MSE_Loss() #MSE_Loss(lmbda=args.lmbda)
    else:
        criterion = MS_SSIM_Loss(device) #(device, lmbda=args.lmbda)
    last_epoch = 0
    best_loss = float("inf")
    # 如果指定了预训练模型,则加载参数。
    if args.i_model_path:  #load from previous checkpoint
        print("Loading model: ", args.i_model_path)
        checkpoint = torch.load(args.i_model_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])   
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_b_model_path = os.path.join(os.path.split(args.i_model_path)[0], 'ckpt.best.pth.tar')
        best_loss = torch.load(best_b_model_path)["loss"]


    log_dir, experiment_id = get_output_folder('./checkpoints/{}/{}/{}/lamda{}/'.format(args.data_name, args.metric, args.model_name, int(args.lmbda)), 'train')
    display_name = "{}_{}_lmbda{}".format(args.model_name, args.metric, int(args.lmbda))
    tags = "lmbda{}".format(args.lmbda)

    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    # 设置输出目录,配置wandb for 训练日志。
    project_name = "VBMIC_" + args.data_name
    wandb.init(project=project_name, name=display_name, tags=[tags],) #notes="lmbda{}".format(args.lmbda))
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    wandb.config.update(args) # config is a variable that holds and saves hyper parameters and inputs
  
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"
    # 主训练循环:
    #val_loss = test_epoch(0, test_dataloader, net, criterion, args)
    for epoch in range(last_epoch, args.epochs):
        #adjust_learning_rate(optimizer, aux_optimizer, epoch, args)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, args)
        lr_scheduler.step()

        wandb.log({"train": {"loss": train_loss["loss"], metric_name: train_loss[metric_name], "bpp_loss": train_loss["bpp_loss"],
            "aux_loss": train_loss["aux_loss"], metric_dB_name: train_loss[metric_dB_name], "left_bpp": train_loss["left_bpp"], "right_bpp": train_loss["right_bpp"],
            left_db_name:train_loss[left_db_name], right_db_name: train_loss[right_db_name]}, }
        )
        if epoch%10==0:
            val_loss = test_epoch(epoch, test_dataloader, net, criterion, args)
            wandb.log({ 
                "test": {"loss": val_loss["loss"], metric_name: val_loss[metric_name], "bpp_loss": val_loss["bpp_loss"],
                "aux_loss": val_loss["aux_loss"], metric_dB_name: val_loss[metric_dB_name], "left_bpp": val_loss["left_bpp"], "right_bpp": val_loss["right_bpp"],
                left_db_name:val_loss[left_db_name], right_db_name: val_loss[right_db_name],}
                })
        
            loss = val_loss["loss"]
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        else:
            loss = best_loss
            is_best = False
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                is_best, log_dir
            )

if __name__ == "__main__":
    main(sys.argv[1:])