"""
3D U-Net diffusion model for WRF data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class TimeEmbedding(nn.Module):
    """Time embedding for diffusion models"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Create sinusoidal embedding
        self.freqs = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000) / embedding_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # t: [batch_size]
        device = t.device
        
        # Create frequency embeddings
        freqs = self.freqs.to(device)
        
        # Compute embeddings
        embeddings = torch.zeros(t.shape[0], self.embedding_dim, device=device)
        embeddings[:, 0::2] = torch.sin(t[:, None] * freqs)
        embeddings[:, 1::2] = torch.cos(t[:, None] * freqs)
        
        return embeddings


class ConvBlock3D(nn.Module):
    """3D convolutional block with residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 time_embed_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Time embedding projection
        self.time_embed_proj = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Residual connection
        self.residual_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x: [batch_size, channels, time, height, width]
        # t_embed: [batch_size, time_embed_dim]
        
        # Initial convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        t_proj = self.time_embed_proj(t_embed)
        h = h + t_proj[:, :, None, None, None]
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        # Residual connection
        return h + self.residual_conv(x)


class AttentionBlock3D(nn.Module):
    """3D attention block"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv_proj = nn.Conv3d(channels, channels * 3, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, T, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # QKV projection
        qkv = self.qkv_proj(h)  # [B, 3*C, T, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # [B, C, T, H, W] each
        
        # Reshape for attention
        q = q.view(B, self.num_heads, C // self.num_heads, T, H, W)
        k = k.view(B, self.num_heads, C // self.num_heads, T, H, W)
        v = v.view(B, self.num_heads, C // self.num_heads, T, H, W)
        
        # Flatten spatial dimensions
        q = q.flatten(3)  # [B, num_heads, C//num_heads, T*H*W]
        k = k.flatten(3)  # [B, num_heads, C//num_heads, T*H*W]
        v = v.flatten(3)  # [B, num_heads, C//num_heads, T*H*W]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, num_heads, T*H*W, T*H*W]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B, num_heads, C//num_heads, T*H*W]
        
        # Reshape back
        out = out.view(B, self.num_heads, C // self.num_heads, T, H, W)
        out = out.contiguous().view(B, C, T, H, W)
        
        # Output projection
        out = self.out_proj(out)
        
        return x + out


class Downsample3D(nn.Module):
    """3D downsampling block"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling block"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample using interpolation
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for diffusion model"""
    
    def __init__(self, 
                 in_channels: int = 17,
                 out_channels: int = 17,
                 channels: List[int] = [64, 128, 256, 512],
                 attention_levels: List[int] = [2, 3],
                 time_embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, channels[0], 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(len(channels)):
            in_ch = channels[i-1] if i > 0 else channels[0]
            out_ch = channels[i]
            
            # Convolutional blocks
            blocks = nn.ModuleList([
                ConvBlock3D(in_ch, out_ch, time_embed_dim, dropout)
            ])
            
            # Add attention if specified
            if i + 1 in attention_levels:
                blocks.append(AttentionBlock3D(out_ch))
            
            self.encoder.append(blocks)
            
            # Downsample (except last level)
            if i < len(channels) - 1:
                self.downsample_layers.append(Downsample3D(out_ch))
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ConvBlock3D(channels[-1], channels[-1], time_embed_dim, dropout),
            AttentionBlock3D(channels[-1]),
            ConvBlock3D(channels[-1], channels[-1], time_embed_dim, dropout)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(len(channels) - 2, -1, -1):
            in_ch = channels[i + 1]
            out_ch = channels[i]
            
            # Upsample
            self.upsample_layers.append(Upsample3D(in_ch))
            
            # Convolutional blocks
            blocks = nn.ModuleList([
                ConvBlock3D(in_ch + out_ch, out_ch, time_embed_dim, dropout)
            ])
            
            # Add attention if specified
            if i + 1 in attention_levels:
                blocks.append(AttentionBlock3D(out_ch))
            
            self.decoder.append(blocks)
        
        # Final output
        self.final_norm = nn.GroupNorm(8, channels[0])
        self.final_conv = nn.Conv3d(channels[0], out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x: [batch_size, channels, time, height, width]
        # t: [batch_size]
        
        # Time embedding
        t_embed = self.time_embed(t)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder
        encoder_outputs = []
        for i, (blocks, downsample) in enumerate(zip(self.encoder, self.downsample_layers)):
            # Apply blocks
            for block in blocks:
                h = block(h, t_embed)
            
            # Store for skip connection
            encoder_outputs.append(h)
            
            # Downsample
            h = downsample(h)
        
        # Middle block
        for block in self.middle_block:
            h = block(h, t_embed)
        
        # Decoder
        for i, (blocks, upsample) in enumerate(zip(self.decoder, self.upsample_layers)):
            # Upsample
            h = upsample(h)
            
            # Skip connection
            skip = encoder_outputs[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            
            # Apply blocks
            for block in blocks:
                h = block(h, t_embed)
        
        # Final output
        h = self.final_norm(h)
        h = F.silu(h)
        output = self.final_conv(h)
        
        return output


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process for WRF data"""
    
    def __init__(self, 
                 model: nn.Module,
                 num_steps: int = 1000,
                 beta_schedule: str = "linear",
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        super().__init__()
        
        self.model = model
        self.num_steps = num_steps
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == "cosine":
            # Cosine schedule
            steps = torch.arange(num_steps + 1) / num_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = torch.clamp(betas, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion process"""
        # x_0: [batch_size, channels, time, height, width]
        # t: [batch_size]
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise to x_0
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        return x_t, noise
    
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion process (denoising)"""
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Get coefficients
        beta_t = self.betas[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        # Denoise
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_t
        
        return x_0_pred
    
    def sample(self, x_t: torch.Tensor, t_start: int = None) -> torch.Tensor:
        """Sample from the model"""
        if t_start is None:
            t_start = self.num_steps - 1
        
        # Iterative denoising
        x_t = x_t.clone()
        
        for t in range(t_start, -1, -1):
            t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
            
            # Predict x_0
            x_0_pred = self.reverse_process(x_t, t_tensor)
            
            if t > 0:
                # Sample x_{t-1}
                noise = torch.randn_like(x_t)
                beta_t = self.betas[t]
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                
                x_t = torch.sqrt(alpha_t) * x_0_pred + torch.sqrt(1 - alpha_t) * noise
            else:
                x_t = x_0_pred
        
        return x_t
    
    def compute_loss(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute training loss"""
        # Forward diffusion
        x_t, noise = self.forward(x_0, t)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


class WRFDiffusionModel(nn.Module):
    """Complete WRF diffusion model"""
    
    def __init__(self, config):
        super().__init__()
        
        # Create U-Net model
        self.unet = UNet3D(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            channels=config.model.channels,
            attention_levels=config.model.attention_levels,
            time_embed_dim=config.model.time_embed_dim
        )
        
        # Create diffusion process
        self.diffusion = GaussianDiffusion(
            model=self.unet,
            num_steps=config.model.num_steps,
            beta_schedule=config.model.beta_schedule,
            beta_start=config.model.beta_start,
            beta_end=config.model.beta_end
        )
        
        self.config = config
    
    def forward(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.diffusion.compute_loss(x_0, t)
    
    def sample(self, x_t: torch.Tensor, t_start: int = None) -> torch.Tensor:
        """Sample from the model"""
        return self.diffusion.sample(x_t, t_start)
    
    def predict_next_step(self, current_state: torch.Tensor, 
                         num_steps: int = 50) -> torch.Tensor:
        """Predict next time step"""
        # Add noise
        t_start = num_steps
        noise = torch.randn_like(current_state)
        
        # Forward diffusion
        sqrt_alpha_cumprod = self.diffusion.sqrt_alphas_cumprod[t_start].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod[t_start].view(-1, 1, 1, 1, 1)
        
        x_t = sqrt_alpha_cumprod * current_state + sqrt_one_minus_alpha_cumprod * noise
        
        # Reverse diffusion
        prediction = self.sample(x_t, t_start)
        
        return prediction