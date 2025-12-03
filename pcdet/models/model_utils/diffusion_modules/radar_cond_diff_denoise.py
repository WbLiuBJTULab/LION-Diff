# change from V2X-R project V2X-R-master\opencood\models\mdd_modules\radar_cond_diff_denoise.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import time
from .diffusion_utils import default, extract_into_tensor, make_beta_schedule, noise_like
import os

from .unet import DiffusionUNet

INTERPOLATE_MODE = 'bilinear'


def tolist(a):
    try:
        return [tolist(i) for i in a]
    except TypeError:
        return a


class Config:
    def __init__(self, entries: dict = {}):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v


class Cond_Diff_Denoise(nn.Module):
    def __init__(self, model_cfg, embed_dim=32):
        super().__init__()
        ### hyper-parameters
        # _代码修改：调整为预测噪声的方式
        self.parameterization = 'eps'
        # self.parameterization = 'x0'
        # beta_schedule = "linear"
        config = Config(model_cfg)

        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        beta_schedule = config.diffusion.beta_schedule
        timesteps = config.diffusion.num_diffusion_timesteps

        # linear_start = 5e-3
        # linear_end = 5e-2

        linear_start = config.diffusion.linear_start
        linear_end = config.diffusion.linear_end

        self.v_posterior = v_posterior = 0
        # 新增loss配置
        self.loss_type = "l2"  # 可以使用'l1', 'l2', 'smooth_l1'
        # self.signal_scaling_rate = 1
        ###

        new_embed_dim = embed_dim

        self.denoiser = DiffusionUNet(config.model)

        self.debug_prefix = False
        self.use_pos_guide = model_cfg.model.use_pos_guide
        self.pos_guide_strength = model_cfg.model.pos_guide_strength

        # q sampling
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # diffusion loss
        learn_logvar = False
        logvar_init = 0
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        self.l_simple_weight = 1.

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        if len(lvlb_weights) > 1:
            lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def compute_diff_loss(self, predicted_noise, true_noise, loss_type=None):
        """
        计算扩散模型loss（预测噪声与真实噪声的差异）

        参数:
            predicted_noise: 模型预测的噪声 [N, latent_dim]
            true_noise: 真实添加的噪声 [N, latent_dim]
            loss_type: 损失类型，默认为self.loss_type

        返回:
            loss: 标量损失值
        """
        if loss_type is None:
            loss_type = self.loss_type

        if loss_type == "l2":
            loss = F.mse_loss(predicted_noise, true_noise, reduction='mean')
        elif loss_type == "l1":
            loss = F.l1_loss(predicted_noise, true_noise, reduction='mean')
        elif loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(predicted_noise, true_noise, reduction='mean')
        else:
            raise ValueError(f"不支持的loss类型: {loss_type}")

        return loss

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, feat, upsam, noisy_masks, t, clip_denoised: bool, coords=None):
        """修改：增加coords参数传递"""
        model_out = self.gen_pred(feat, noisy_masks, coords=coords, upsam=upsam, t=t)

        x = noisy_masks
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        if upsam:  # 最后一步采样
            model_mean = x_recon
            posterior_variance, posterior_log_variance = 0, 0
        else:
            # 点级数据无需插值：移除F.interpolate调用
            # 原代码：x_recon = F.interpolate(x_recon, x.shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        assert model_mean.shape == x_recon.shape
        return model_mean, posterior_variance, posterior_log_variance, model_out

    def p_sample(self, feat, noisy_masks, t, upsam, clip_denoised=False, repeat_noise=False, coords=None):
        """修改：增加coords参数传递"""
        model_mean, _, model_log_variance, model_out = self.p_mean_variance(
            feat, upsam, noisy_masks, t=t, clip_denoised=clip_denoised, coords=coords)

        x = noisy_masks
        b, *_, device = *x.shape, x.device
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if not upsam:
            out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise  # , model_out
        else:
            out = model_mean

        return out, model_out

    def gen_pred(self, feat, noisy_masks, coords=None, upsam=False, t=None):
        """生成预测，传递坐标到Unet"""
        x_input = torch.cat([feat, noisy_masks], dim=1)

        # 关键修改：根据是否启用位置引导传递坐标
        if self.use_pos_guide and coords is not None:
            # 坐标预处理：去掉batch维度，只保留空间坐标
            debs_coords = coords[:, 1:]  # [N, 3] (z, y, x)
            model_out = self.denoiser(x_input, t.float(), coords=debs_coords)
        else:
            model_out = self.denoiser(x_input, t.float(), coords=None)

        return model_out

    def p_sample_loop(self, feat, noisy_masks, latent_shape, coords=None):
        """修改：增加coords参数传递"""
        # feat 相当于去噪对象
        # noisy_masks 相当于噪声数据
        b = latent_shape[0]
        num_timesteps = self.num_timesteps

        for t in reversed(range(0, num_timesteps)):
            noisy_masks, _ = self.p_sample(
                feat, noisy_masks,
                torch.full((b,), t, device=feat.device, dtype=torch.long),
                upsam=True if t == 0 else False,
                coords=coords  # 传递坐标信息
            )
        return noisy_masks

    # _20250814重要代码：添加diffusion model的generate方法
    # === 新增特征生成方法 ===
    def generate(self, conditions, num_points):
        """生成扩散特征

        参数:
            conditions: 条件特征 [K, C]
            num_points: 每个条件生成的点数

        返回:
            生成的特征 [K*num_points, C]
        """
        # 1. 准备噪声输入 [K*num_points, C]
        noise = torch.randn(num_points * len(conditions),
                            conditions.shape[1],
                            device=conditions.device)

        # 2. 扩散生成过程
        generated = self.p_sample_loop(
            feat=conditions.unsqueeze(0),  # 添加batch维度
            noisy_masks=noise.unsqueeze(0),  # 添加batch维度
            latent_shape=(1, conditions.shape[1])
        )
        return generated.squeeze(0)  # [K*num_points, C]

    def forward(self, data_dict):

        # _代码修改：原始为将低质量Radar特征增强为LiDAR级别
        # 先为将体素地图latent增强为真值框体素蒙版地图latent

        if self.training:
            gt_mask_latent = data_dict['gt_mask_latent']
            bs_voxel_latent = data_dict['bs_voxel_latent']

            total_loss = 0
            step_count = 0

            # 组合坐标信息：使用蒙版坐标和原始坐标
            if self.use_pos_guide:
                # 关键修改：直接拼接，保持[N,4]形状
                coords_condition = data_dict['gt_mask_coords']
            else:
                coords_condition = None

            if self.debug_prefix:
                print(f"Cond_Diff_Denoise相关输入-gt_mask_latent", gt_mask_latent.shape)
                print(f"Cond_Diff_Denoise相关输入-bs_voxel_latent", bs_voxel_latent.shape)

            x = bs_voxel_latent
            x_start = gt_mask_latent

            if self.debug_prefix:
                print(f"    -扩散模型流程：x", x.shape)
                print(f"    -扩散模型流程：x_start", x_start.shape)

            t = torch.full((x.shape[0],), self.num_timesteps - 1, device=x.device, dtype=torch.long)
            noise = default(None, lambda: torch.randn_like(x_start))
            true_noise = noise

            if self.debug_prefix:
                print(f"    -扩散模型流程：t", t.shape)
                print(f"    -扩散模型流程：noise", noise.shape)

            _x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            noisy_masks = _x_noisy  # 此处出了训练用的噪声真值

            # 修改训练循环，收集每一步的预测噪声
            # 注意这里的噪声处理是每一次调度都经过了完整的噪声训练过程，但由于我们仅使用无不扩散，所以这样还能接受
            predicted_noises = []
            for _t in reversed(range(1, self.num_timesteps)):
                _t = torch.full((x.shape[0],), _t, device=x.device, dtype=torch.long)
                noisy_masks, pred_noise = self.p_sample(x, noisy_masks, _t, upsam=False, coords=coords_condition)
                predicted_noises.append(pred_noise)
                step_count += 1

            _t = 0
            _t = torch.full((x.shape[0],), _t, device=x.device, dtype=torch.long)
            noisy_masks, pred_noise = self.p_sample(x, noisy_masks, _t, upsam=True, coords=coords_condition)
            predicted_noises.append(pred_noise)
            step_count += 1

            total_loss = sum([self.compute_diff_loss(pred, true_noise)
                              for pred in predicted_noises])
            loss = total_loss / step_count

            enhanced_features = noisy_masks

            if self.debug_prefix:
                print(f"    -扩散模型流程：enhanced_features", enhanced_features.shape)

            return enhanced_features, loss

        else:
            bs_voxel_latent = data_dict['bs_voxel_latent']

            # 组合坐标信息：使用蒙版坐标和原始坐标
            if self.use_pos_guide:
                # 关键修改：直接拼接，保持[N,4]形状
                coords_condition = data_dict['bs_voxel_coords']
            else:
                coords_condition = None

            t1 = time.time()
            x = bs_voxel_latent  # 条件特征 [N, 16]
            # 初始化noisy_masks为随机噪声，形状与voxel_latent相同
            noisy_masks = torch.randn_like(x)  # 标准扩散模型初始化
            t2 = time.time()
            # 直接进行去噪循环，条件为voxel_latent
            noisy_masks = self.p_sample_loop(x, noisy_masks, x.shape, coords=coords_condition)
            t3 = time.time()
            enhanced_features = noisy_masks # 最终输出的去噪后的数据分布

            return enhanced_features, 0.0
