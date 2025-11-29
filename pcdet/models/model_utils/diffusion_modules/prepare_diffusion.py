"""
扩散模型工具函数库
包含与扩散模型相关的坐标处理、特征提取和条件生成功能
"""

import torch
import torch.nn as nn
import math
import numpy as np

from .radar_cond_diff_denoise import Cond_Diff_Denoise


class DiffusionCoordinateProcessor:
    """
    处理扩散模型相关的坐标操作
    包括坐标下采样、坐标扩散生成、坐标验证等
    """

    @staticmethod
    def downsample_coords(coords, down_scale, spatial_shape):
        """
        下采样坐标张量
        参数:
            coords: 坐标张量 [N, 4] (batch_idx, z, y, x)
            down_scale: 下采样比例 [scale_x, scale_y, scale_z]
            spatial_shape: 空间形状 (D, H, W)
        返回:
            下采样后的坐标张量
        """
        if coords is None or len(coords) == 0:
            return None

        downsampled_coords = coords.clone().float()
        downsampled_coords[:, 3] = torch.floor(downsampled_coords[:, 3] / down_scale[0])  # X
        downsampled_coords[:, 2] = torch.floor(downsampled_coords[:, 2] / down_scale[1])  # Y
        downsampled_coords[:, 1] = torch.floor(downsampled_coords[:, 1] / down_scale[2])  # Z

        # 裁剪到有效范围
        d, h, w = spatial_shape
        downsampled_coords[:, 3] = downsampled_coords[:, 3].clamp(0, w - 1)
        downsampled_coords[:, 2] = downsampled_coords[:, 2].clamp(0, h - 1)
        downsampled_coords[:, 1] = downsampled_coords[:, 1].clamp(0, d - 1)

        return downsampled_coords.long()

    @staticmethod
    def generate_diffusion_coords(original_coords, num_points, coords_shift, spatial_shape):
        """
        基于扩散模型学习到的分布生成新坐标
        参数:
            original_coords: 原始坐标 [N, 4]
            num_points: 每个坐标生成的点数
            coords_shift: 坐标偏移量
            spatial_shape: 空间形状 (D, H, W)
        返回:
            生成的坐标张量
        """
        d, h, w = spatial_shape

        # 扩展原始坐标
        expanded_coords = original_coords.repeat(num_points, 1)

        # 生成随机偏移（基于学习到的分布）
        batch_size = expanded_coords[:, 0].max() + 1
        generated_coords_list = []

        for i in range(batch_size):
            batch_mask = expanded_coords[:, 0] == i
            if not batch_mask.any():
                continue

            batch_coords = expanded_coords[batch_mask]
            N_batch = len(batch_coords)

            # 生成随机偏移（正态分布）
            shift_z = torch.randn(N_batch, 1, device=original_coords.device) * coords_shift
            shift_y = torch.randn(N_batch, 1, device=original_coords.device) * coords_shift
            shift_x = torch.randn(N_batch, 1, device=original_coords.device) * coords_shift

            new_coords = batch_coords.clone()
            new_coords[:, 1:2] = (new_coords[:, 1:2] + shift_z).clamp(0, d - 1)  # Z
            new_coords[:, 2:3] = (new_coords[:, 2:3] + shift_y).clamp(0, h - 1)  # Y
            new_coords[:, 3:4] = (new_coords[:, 3:4] + shift_x).clamp(0, w - 1)  # X

            generated_coords_list.append(new_coords)

        return torch.cat(generated_coords_list) if generated_coords_list else torch.tensor([],
                                                                                           device=original_coords.device)

    @staticmethod
    def validate_coordinate_extraction(matched_coords, reference_coords, spatial_shape):
        """
        验证坐标提取结果的准确性
        参数:
            matched_coords: 匹配的坐标
            reference_coords: 参考坐标
            spatial_shape: 空间形状
        """
        d, h, w = spatial_shape

        # 验证所有匹配坐标确实在参考坐标中
        matched_linear = (matched_coords[:, 0] * (d * h * w) +
                          matched_coords[:, 1] * (h * w) +
                          matched_coords[:, 2] * w +
                          matched_coords[:, 3])

        ref_linear = (reference_coords[:, 0] * (d * h * w) +
                      reference_coords[:, 1] * (h * w) +
                      reference_coords[:, 2] * w +
                      reference_coords[:, 3])

        ref_set = set(ref_linear.cpu().numpy())
        for key in matched_linear:
            assert key.item() in ref_set, f" 扩散条件准备: 提取的坐标{key.item()}不在参考坐标中"

        print(f"[DEBUG] 扩散条件准备: 验证通过 - 所有匹配坐标均在参考坐标范围内")

class DiffusionFeatureExtractor:
    """
    处理扩散模型相关的特征提取操作
    包括真值框特征提取、特征融合、潜变量编码等
    """

    @staticmethod
    def extract_gt_reference_features(bs_voxel_coords, bs_voxel_features, reference_coords,
                                      spatial_shape, device, debug_prefix=False):
        """
        提取真值框参考地图范围内的体素特征，并返回蒙版信息。
        参数:
            x: 稀疏张量 (spconv.SparseConvTensor)
            reference_coords: 参考坐标 [N, 4]
            return_full_mask: 是否返回完整的蒙版信息（包括冗余输出）
        返回:
            masked_features: 蒙版体素特征地图（真值框内的特征）
            masked_coords: 蒙版体素地图坐标（真值框内的坐标）
            gt_features: 真值框体素特征地图（与 masked_features 相同）
            gt_coords: 真值框体素地图坐标（与 masked_coords 相同）
            注意：masked_* 和 gt_* 是相同的，返回四元组以兼容需求。
        """
        # 输入验证
        if reference_coords is None or len(reference_coords) == 0:
            if debug_prefix:
                print(f"[DEBUG] 扩散条件准备: 参考坐标为空，跳过特征提取")
            return None, None, None, None  # 返回四个None


        # 设备一致性处理
        reference_coords = reference_coords.to(device)

        # 获取空间维度信息
        d, h, w = spatial_shape

        # 创建坐标线性化映射函数
        def linearize_coords(coords):
            return (coords[:, 0] * (d * h * w) +
                    coords[:, 1] * (h * w) +
                    coords[:, 2] * w +
                    coords[:, 3])

        # 线性化参考坐标
        ref_linear = linearize_coords(reference_coords)
        ref_set = set(ref_linear.cpu().numpy())

        # 线性化当前体素坐标
        x_linear = linearize_coords(bs_voxel_coords)

        # 创建匹配掩码
        matched_mask = torch.tensor([key.item() in ref_set for key in x_linear],
                            device=device, dtype=torch.bool)

        # 提取匹配结果
        matched_features = bs_voxel_features[matched_mask]
        matched_coords = bs_voxel_coords[matched_mask]


        # 关键改进：创建蒙版体素特征地图
        # 创建蒙版特征地图：真值框内为原特征，框外为零
        mask_features = torch.zeros_like(bs_voxel_features)  # 初始化为零
        mask_coords = torch.zeros_like(bs_voxel_coords)  # 初始化为零
        mask_features[matched_mask] = bs_voxel_features[matched_mask]  # 真值框内填入实际特征
        mask_coords[matched_mask] = bs_voxel_coords[matched_mask]  # 真值框内填入实际特征

        # 创建蒙版标记
        mask_indicator = matched_mask.clone()

        if debug_prefix:
            print(f"[DEBUG] 扩散条件准备: 蒙版地图创建完成")
            print(f"        - 真值框特征形状: {matched_features.shape}")
            print(f"        - 真值框坐标形状: {matched_coords.shape}")
            print(f"        - 蒙版特征形状: {mask_features.shape}")
            print(f"        - 蒙版坐标形状: {mask_coords.shape}")
            print(f"        - 蒙版标记形状: {mask_indicator.shape}")
            print(f"        - 蒙版标记True数量: {mask_indicator.sum().item()}")

            # 验证提取准确性
            DiffusionCoordinateProcessor.validate_coordinate_extraction(
                matched_coords, reference_coords, spatial_shape)

        return matched_features, matched_coords, mask_features, mask_coords, mask_indicator

    @staticmethod
    def prepare_diffusion_input(bs_voxel_features, bs_voxel_coords, reference_coords, spatial_shape, device):
        """
        准备扩散模型的输入条件
        参数:
            x: 稀疏张量
            fill_coords: 填充坐标
            reference_coords: 参考坐标
            diffusion_model_instance: 扩散模型实例
        返回:
            扩散条件字典
        """
        diffusion_input = {}

        feature_dim = bs_voxel_features.shape[1] if len(bs_voxel_features) > 0 else 64
        coord_dim = 4  # (batch_idx, z, y, x)

        # 提取真值框特征作为条件
        if reference_coords is not None and len(reference_coords) > 0:
            gt_reference_features, gt_reference_coords, mask_features, mask_coords, mask_indicator = (
                DiffusionFeatureExtractor.extract_gt_reference_features(
                bs_voxel_coords, bs_voxel_features, reference_coords, spatial_shape, device)
            )
            diffusion_input.update({
                'bs_voxel_features': bs_voxel_features,
                'bs_voxel_coords': bs_voxel_coords,
                'gt_reference_features': gt_reference_features,
                'gt_reference_coords': gt_reference_coords,
                'gt_mask_features': mask_features,  # 蒙版特征地图
                'gt_mask_coords': mask_coords,
                'gt_mask_indicator': mask_indicator,  # 蒙版标记
                'has_gt_reference': True
            })
        else:
            # 处理空reference_coords的情况
            empty_features = torch.empty(0, feature_dim, device=device)
            empty_coords = torch.empty(0, coord_dim, device=device, dtype=torch.long)
            zero_features = torch.zeros_like(bs_voxel_features)
            zero_coords = torch.zeros_like(bs_voxel_coords)
            mask_indicator = torch.zeros(len(bs_voxel_features), dtype=torch.bool, device=device)

            diffusion_input.update({
                'bs_voxel_features': bs_voxel_features,
                'bs_voxel_coords': bs_voxel_coords,
                'gt_reference_features': empty_features,
                'gt_reference_coords': empty_coords,
                'gt_mask_features': zero_features,
                'gt_mask_coords': zero_coords,
                'gt_mask_indicator': mask_indicator,
                'has_gt_reference': False
            })
            # print("[DEBUG] 扩散条件准备: reference_coords 为空，使用全零负样本")

        return diffusion_input


class VoxelLatentEncoder(nn.Module):
    """
    增强版编码器：使用深层MLP将体素特征和坐标映射到潜空间。
    输入: features [N, C],
    输出: latent [N, D], D=8-16
    """

    def __init__(self, input_dim=64, latent_dim=16):
        super().__init__()
        self.voxel_to_latent_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)  # 坐标编码为16维
        )

    def forward(self, features):
        latent = self.voxel_to_latent_encoder(features)  # [N, latent_dim]
        return latent


class VoxelLatentDecoder(nn.Module):
    """
    潜变量解码器：将扩散模型输出的潜变量解码回原始体素特征维度
    输入: latent [N, 16]
    输出: features [N, 64]
    """

    def __init__(self, latent_dim=16, output_dim=64):
        super().__init__()
        self.latent_to_voxel_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)  # 解码回64维体素特征
        )

    def forward(self, latent):
        return self.latent_to_voxel_decoder(latent)

class DiffusionModelManager:
    """
    管理扩散模型的训练流程
    包括训练/推断模式切换、梯度控制、损失计算等
    """

    def __init__(self, diff_model):

        self.latent_encoder_instance = VoxelLatentEncoder(
            input_dim=diff_model.LATENT.voxel_feature_dim,
            latent_dim=diff_model.LATENT.voxel_latent_dim
        )

        # 新增解码器
        self.latent_decoder_instance = VoxelLatentDecoder(
            latent_dim=diff_model.LATENT.voxel_latent_dim,
            output_dim=diff_model.LATENT.voxel_feature_dim
        )

        self.diffusion_model_instance = Cond_Diff_Denoise(
            model_cfg=diff_model.DIFF_MODEL_CFG,
            embed_dim=diff_model.LATENT.voxel_feature_dim
        )

        # self.diff_noise_scale = diff_model.DIFF_PROCESS.diff_noise_scale

        self._device_synced = False

        self.debug_prefix = False

    def _sync_device(self, input_tensor):
        """同步模型设备到输入张量所在的设备"""
        if not self._device_synced:
            target_device = input_tensor.device
            self.latent_encoder_instance = self.latent_encoder_instance.to(target_device)
            self.latent_decoder_instance = self.latent_decoder_instance.to(target_device)  # 新增解码器同步
            self.diffusion_model_instance = self.diffusion_model_instance.to(target_device)
            self._device_synced = True

    def apply_diffusion(self, diffusion_input, training=True):
        """
        应用扩散模型
        参数:
            diffusion_input: 条件输入
            target: 目标特征（用于训练）
        返回:
            增强后的特征
        """
        if 'bs_voxel_features' in diffusion_input:
            self._sync_device(diffusion_input['bs_voxel_features'])

        if training:
            # 训练模式
            if self.debug_prefix:
                print(f"[DEBUG] 已进入扩散训练流程")
                print(f"扩散输入设备: {diffusion_input['bs_voxel_features'].device}")
                print(f"编码器设备: {next(self.latent_encoder_instance.parameters()).device}")
                print(f"扩散模型设备: {next(self.diffusion_model_instance.parameters()).device}")
                print(f"解码器设备: {next(self.latent_decoder_instance.parameters()).device}")

            # _代码修改：同时对体素地图特征和真值框参考蒙版地图基于MLP提取latent
            voxel_latent = self.latent_encoder_instance(diffusion_input['bs_voxel_features'])
            gt_mask_latent = self.latent_encoder_instance(diffusion_input['gt_mask_features'])
            diffusion_input.update({
                'voxel_latent': voxel_latent,
                'gt_mask_latent': gt_mask_latent
            })

            if self.debug_prefix:
                print(f"潜变量形状: {diffusion_input['voxel_latent'].shape}")

            enhanced_latent, diffusion_loss = self.diffusion_model_instance(diffusion_input)

            # enhanced_latent = voxel_latent - predicted_noise * self.diff_noise_scale
            # enhanced_features = self.latent_decoder_instance(enhanced_latent)
            enhanced_voxel = self.latent_decoder_instance(enhanced_latent)

            return enhanced_voxel, diffusion_loss

        else:
            # 推断模式
            with torch.no_grad():
                if self.debug_prefix:
                    print(f"[DEBUG] 已进入扩散测试流程")
                    print(f"扩散输入设备: {diffusion_input['bs_voxel_features'].device}")
                    print(f"编码器设备: {next(self.latent_encoder_instance.parameters()).device}")
                    print(f"扩散模型设备: {next(self.diffusion_model_instance.parameters()).device}")

                voxel_latent = self.latent_encoder_instance(diffusion_input['bs_voxel_features'])
                gt_mask_latent = self.latent_encoder_instance(diffusion_input['gt_mask_features'])
                diffusion_input.update({
                    'voxel_latent': voxel_latent,
                    'gt_mask_latent': gt_mask_latent
                })

                if self.debug_prefix:
                    print(f"潜变量形状: {diffusion_input['voxel_latent'].shape}")

                #enhanced_features = diffusion_input['voxel_features']
                enhanced_latent, _ = self.diffusion_model_instance(diffusion_input)

                # enhanced_latent = voxel_latent - predicted_noise * self.diff_noise_scale
                # enhanced_features = self.latent_decoder_instance(enhanced_latent)
                enhanced_voxel = self.latent_decoder_instance(enhanced_latent)

            return enhanced_voxel, 0.0
