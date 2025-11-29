import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiffusionUNet(nn.Module):
    """
    简化版DiffusionUNet：使用坐标编码加法融合替代复杂注意力。
    输入: [N, 32] (feat 16 + noisy_masks 16)
    输出: [N, 16] (去噪后的潜在特征)
    """
    def __init__(self, config):
        super().__init__()
        # 现有参数
        self.in_channels = config.in_channels * 2  # 32
        self.out_channels = config.out_channels  # 16
        self.base_channels = config.base_channels  # 32
        self.dropout = config.dropout
        self.use_pos_guide = config.use_pos_guide
        self.pos_embed_dim = config.pos_embed_dim
        self.temb_ch = self.base_channels * 4  # 128

        # 时间步嵌入网络（保持不变）
        self.temb_net = nn.Sequential(
            nn.Linear(self.base_channels, self.temb_ch),
            nn.SiLU(),
            nn.Linear(self.temb_ch, self.temb_ch)
        )

        # 简化位置引导：坐标编码器和投影层
        if self.use_pos_guide:
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, self.pos_embed_dim),  # 输入坐标(z,y,x)
                nn.ReLU(),
                nn.Linear(self.pos_embed_dim, self.pos_embed_dim)
            )
            # 投影层：将坐标编码映射到特征维度（160 = 32 + 128）
            self.pos_proj = nn.Linear(self.pos_embed_dim, self.in_channels + self.temb_ch)
        else:
            self.pos_encoder = None
            self.pos_proj = None

        # MLP主干网络（保持不变，但移除注意力相关部分）
        if hasattr(config, 'mlp_dims'):
            mlp_dims = config.mlp_dims
        else:
            mlp_dims = [256, 128, 64]  # 默认值

        layers = []
        input_dim = self.in_channels + self.temb_ch  # 当前为 32+128=160
        for dim in mlp_dims:
            layers.append(nn.Sequential(
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.SiLU(),
                nn.Dropout(self.dropout)
            ))
            input_dim = dim
        layers.append(nn.Linear(input_dim, self.out_channels))
        self.mlp_layers = nn.ModuleList(layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, t, coords=None):
        # 时间步处理（保持不变）
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.size(0) == 1 and x.size(0) > 1:
            t = t.repeat(x.size(0))
        temb = get_timestep_embedding(t, self.base_channels)
        temb = self.temb_net(temb)
        if x.size(0) != temb.size(0):
            if x.size(0) % temb.size(0) == 0:
                repeat_factor = x.size(0) // temb.size(0)
                temb = temb.repeat(repeat_factor, 1)
            else:
                temb = temb.expand(x.size(0), -1)

        # 拼接特征和时间步嵌入
        h = torch.cat([x, temb], dim=1)  # [N, 160]

        # 简化位置引导：加法融合坐标编码
        if self.use_pos_guide and coords is not None:
            coords_float = coords.float()
            pos_embed = self.pos_encoder(coords_float)  # [N, pos_embed_dim]
            pos_embed_proj = self.pos_proj(pos_embed)  # [N, 160]
            h = h + pos_embed_proj  # 加法融合，维度不变

        # 通过MLP层（移除中间的注意力机制）
        for layer in self.mlp_layers:
            h = layer(h)
        return h

def get_timestep_embedding(timesteps, embedding_dim):
    """
    生成时间步嵌入（保持不变）
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)

    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))

    return emb