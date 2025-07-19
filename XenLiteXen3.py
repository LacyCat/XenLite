import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # GroupNorm에 맞는 채널 수 조정
        norm_groups = min(8, out_channels)
        if out_channels % norm_groups != 0:
            norm_groups = out_channels // ((out_channels // 8) + 1)
            
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels) if in_channels >= 8 else 1, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        norm_groups = min(8, channels)
        if channels % norm_groups != 0:
            norm_groups = channels // ((channels // 8) + 1)
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = 1.0 / math.sqrt(C)
        attn = torch.matmul(q.transpose(-2, -1), k) * scale
        attn = F.softmax(attn, dim=-1)
        
        h = torch.matmul(v, attn.transpose(-2, -1))
        h = h.view(B, C, H, W)
        return x + self.proj(h)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, has_attn: bool = False):
        super().__init__()
        self.resnet = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb):
        x = self.resnet(x, time_emb)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, has_attn: bool = False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        # skip connection과 upsampled feature를 합친 채널 수
        self.resnet = ResidualBlock(in_channels + in_channels, out_channels, time_emb_dim)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resnet(x, time_emb)
        x = self.attn(x)
        return x

class XenLiteXen3(nn.Module):
    def __init__(self, in_channels: int = 3, model_channels: int = 48, num_classes: int = 10):
        super().__init__()
        self.model_channels = model_channels
        self.num_classes = num_classes
        
        time_emb_dim = model_channels * 3
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)
        
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 균형잡힌 모델 구조 (3단계 복원)
        self.down1 = DownBlock(model_channels, model_channels, time_emb_dim)
        self.down2 = DownBlock(model_channels, model_channels * 2, time_emb_dim, has_attn=True)
        self.down3 = DownBlock(model_channels * 2, model_channels * 3, time_emb_dim)
        
        self.middle = nn.Sequential(
            ResidualBlock(model_channels * 3, model_channels * 3, time_emb_dim),
            AttentionBlock(model_channels * 3),
            ResidualBlock(model_channels * 3, model_channels * 3, time_emb_dim)
        )
        
        self.up1 = UpBlock(model_channels * 3, model_channels * 2, time_emb_dim)
        self.up2 = UpBlock(model_channels * 2, model_channels, time_emb_dim, has_attn=True)
        self.up3 = UpBlock(model_channels, model_channels, time_emb_dim)
        
        norm_groups = min(6, model_channels)
        if model_channels % norm_groups != 0:
            norm_groups = model_channels // ((model_channels // 6) + 1)
            
        self.output_conv = nn.Sequential(
            nn.GroupNorm(norm_groups, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, class_labels=None):
        time_emb = self.time_embed(timesteps)
        
        if class_labels is not None:
            class_emb = self.class_embed(class_labels)
            time_emb = time_emb + class_emb
        
        x = self.input_conv(x)
        
        x, skip1 = self.down1(x, time_emb)  # 48 -> 48, skip1: 48 channels
        x, skip2 = self.down2(x, time_emb)  # 48 -> 96, skip2: 96 channels
        x, skip3 = self.down3(x, time_emb)  # 96 -> 144, skip3: 144 channels
        
        x = self.middle[0](x, time_emb)  # 144 -> 144
        x = self.middle[1](x)            # 144 -> 144
        x = self.middle[2](x, time_emb)  # 144 -> 144
        
        x = self.up1(x, skip3, time_emb)  # 144 + 144 -> 96
        x = self.up2(x, skip2, time_emb)  # 96 + 96 -> 48
        x = self.up3(x, skip1, time_emb)  # 48 + 48 -> 48
        
        return self.output_conv(x)

class DDPMScheduler:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def sample_prev_timestep(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = (sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output) / self.sqrt_alphas_cumprod[t]
        
        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(self.posterior_variance[t]) * noise
        else:
            variance = 0
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_sample_direction = (1 - alpha_prod_t_prev - torch.sqrt(self.posterior_variance[t])**2) * model_output
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction + variance
        
        return prev_sample

class DDIMScheduler:
    def __init__(self, num_timesteps: int = 1000, num_inference_steps: int = 50):
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        step_ratio = num_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_timesteps, step_ratio).flip(0)

    def sample_prev_timestep(self, model_output, timestep, sample, eta: float = 0.0):
        prev_timestep = timestep - self.num_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (sample - beta_prod_t**(0.5) * model_output) / alpha_prod_t**(0.5)
        
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance**(0.5)
        
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**(0.5) * model_output
        
        prev_sample = alpha_prod_t_prev**(0.5) * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn_like(model_output)
            prev_sample = prev_sample + std_dev_t * noise
            
        return prev_sample