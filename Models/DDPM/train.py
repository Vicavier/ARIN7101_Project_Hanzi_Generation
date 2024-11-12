import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms 
from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import Diffuser
import math
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("./PROJECT"), '../..')))
from Models.Util.loadData import create_hanzi_dataloaders



class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg, use_buffers=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=10)
    parser.add_argument('--model_base_dim', type=int,help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--labels', type=int, help='number of labels for conditional training', default=10)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    # conditional training
    parser.add_argument('--conditional_training', action='store_true', help='conditional training')

    args = parser.parse_args()

    return args

def main(args):
    device = "cpu" if args.cpu else "cuda"
    mode = True if args.conditional_training else False
    preprocess = transforms.Compose([
        transforms.ToTensor(),                       # 转换为 Tensor (C, H, W)，其中 C=1 表示单通道
        transforms.Normalize([0.5], [0.5])           # 单通道灰度图的归一化 [-1, 1]
    ])
    train_dataloader = create_hanzi_dataloaders(root_dir="../../Dataset/hanzi/processed", batch_size=args.batch_size, preprocess=preprocess)
    
    
    model = Diffuser(
        timesteps=args.timesteps,           # DDPM采样步数，默认1000步
        label=args.labels,                  # 条件生成时，指定类别标签
        time_embedding_dim=256,             # 时间嵌入维度
        image_size=32,                      # MNIST数据集图片尺寸为28*28
        in_channels=1,                      # MNIST数据集图片为单通道
        base_dim=args.model_base_dim,       # Unet的最初的特征维度，也就是初始卷积操作后的通道数，默认64
        dim_mults=[2,4,8]                   # 控制Unet中不同层次的通道数，表示每一层的通道数是base_dim的倍数。默认[2,4]
    ).to(device)

    #torchvision ema setting
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(),lr=args.lr)
    scheduler = OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    global_steps=0
    # create project name with current time 
    exp_name = datetime.datetime.now().strftime("%m%d-%H%M%S")
    exp_path = Path("logs") / exp_name
    exp_path.mkdir(parents=True)
    (exp_path / "ckpt").mkdir(exist_ok=True)
    (exp_path / "img").mkdir(exist_ok=True)
    
    ckpt_list = [] #check point
    LOSS = []
    

    for i in range(args.epochs):
        model.train()
        leave_option = False if i < args.epochs - 1 else True
        training_progress = tqdm(train_dataloader, desc='Training Progress', leave=leave_option)

        epoch_loss = 0.0
        num_batches = 0

        for image, label in training_progress:
            noise = torch.randn_like(image).to(device)  # randn_like(image)用于生成与image形状相同的标准正态分布（均值为0，标准差为1）的随机张量。
            image = image.to(device)    # [B,C,H,W]=[256,1,28,28]
            label = label.to(device) if mode else None
            pred = model(image, noise, label)  # [B,C,H,W]=[256,1,28,28]
            loss: Tensor = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            
            epoch_loss += loss.detach().cpu().item()
            num_batches += 1

            training_progress.set_description(f"epoch-{i} loss: {loss.detach().cpu().item():.4f}")
        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }
        ckpt_path = exp_path / "ckpt" / f"{i}.pt"
        torch.save(ckpt, ckpt_path)
        ckpt_list.insert(0, ckpt_path)
        if len(ckpt_list) > 5:
            remove_ckpt = ckpt_list.pop()
            remove_ckpt.unlink()

        avg_epoch_loss = epoch_loss / num_batches
        LOSS.append(avg_epoch_loss)
        model_ema.eval()
        l = None
        if mode:
            l = torch.arange(args.labels).repeat_interleave(args.n_samples//10).to(device)
            for _ in range(args.n_samples - len(l)):
                l = torch.cat((l, torch.randint(0, args.labels, (1,)).to(device)))
                
        samples=model_ema.module.sampling(args.n_samples, device=device,l=l)
        save_image(samples, exp_path / "img" / f"{i}.png", nrow=int(math.sqrt(args.n_samples)))

    with open(f'{exp_path}/losses.txt', 'w') as f:
        for loss in LOSS:
            # 将每个loss值格式化为保留四位小数的字符串，并写入文件
            f.write(f"{loss}\n")

if __name__=="__main__":
    args=parse_args()
    main(args)