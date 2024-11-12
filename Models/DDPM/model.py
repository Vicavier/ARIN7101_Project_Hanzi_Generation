import torch.nn as nn
import torch
from torch import Tensor
import math
from unet import Unet
from tqdm import tqdm
from torchvision.utils import save_image

class Diffuser(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim,timesteps,label,base_dim,dim_mults):
        super().__init__()
        self.timesteps = timesteps
        self.label = label
        self.in_channels = in_channels
        self.image_size = image_size
        self.base_dim = base_dim
        self.dim_mults = dim_mults

        betas = self._cosine_variance_schedule(timesteps)
        # betas = self._linear_variance_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas,dim=-1)   # alphas累乘

        # 注册变量随模型保存和加载
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.-alphas_cumprod))

        self.model = Unet(timesteps,label,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self, x, noise, l):
        # 1.随机选择一个时间步t
        # 2.调用_forward_diffusion方法生成扩散到时间步t的图像x_t
        # 3.使用Unet模型预测噪声pred_noise
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, t, l)
        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples: int, device="cuda",l=None) -> Tensor:
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        images = []
        for i in range(self.timesteps - 1, -1, -1):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            x_t = self._reverse_diffusion_with_clip(x_t, t, noise, l)
            # if(i % 20 == 0):
            #     img_tensor = (x_t + 1.) / 2.
            #     save_image(img_tensor, f'./sampling_progress/step_{i}.png', nrow=3)

        x_t=(x_t + 1.) / 2. #[-1,1] to [0,1]
        
        return x_t
    
    def _cosine_variance_schedule(self, timesteps: int, epsilon=0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
    
    def _linear_variance_schedule(self, timesteps: int):
        '''
            generate cosine variance schedule
            reference: the DDPM paper https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
            You might compare the model performance of linear and cosine variance schedules. 
        '''
        
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        return betas

    def _forward_diffusion(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        '''
            forward diffusion process
            hint: calculate x_t given x_0, t, noise
            please note that alpha related tensors are registered as buffers in __init__, you can use gather method to get the values
            reference: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process
        '''
        # 通过 gather() 函数从缓冲区中提取时间步 t 对应的 sqrt_alpha_t 和 sqrt_one_minus_alpha_t。
        sqrt_alpha_t = self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1)
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t: Tensor, t: Tensor, noise: Tensor, label:Tensor) -> Tensor: 
        '''
            reverse diffusion process with clipping
            hint: with clip: pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
                  without clip: pred_noise -> pred_mean and pred_std
                  you may compare the model performance with and without clipping
        '''
        pred=self.model(x_t,t,label)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    
    @torch.no_grad()
    def _reverse_diffusion_without_clip(self, x_t: Tensor, t: Tensor, noise: Tensor, label: Tensor) -> Tensor: 
        '''
            reverse diffusion process without clipping
            hint: without clip: pred_noise -> pred_mean and pred_std
        '''
        pred = self.model(x_t, t, label)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        
        # Compute x_0_pred without clipping
        x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            mean = (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred + \
                ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t

            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred  # alpha_t_cumprod_prev = 1 since t != 1
            std = 0.0

        return mean + std * noise
    