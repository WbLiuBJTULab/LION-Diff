import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2
from ....utils import box_utils  # 导入box_utils用于点框判断


class DynamicVoxelVFE_ForGtMap(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        # 基本配置
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        # PFN层配置
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # 体素化参数
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def _generate_gt_voxel_maps(self, batch_dict, voxel_coords, points_xyz=None):
        """
        在VFE环节重新生成真值框地图，确保与主流程体素坐标对齐
        """

        if not self.training or 'gt_boxes' not in batch_dict:
            batch_dict['gt_fill_coords'] = torch.zeros((0, 4), dtype=torch.int32, device=batch_dict['points'].device)
            batch_dict['gt_reference_coords'] = torch.zeros((0, 4), dtype=torch.int32,
                                                            device=batch_dict['points'].device)
            return batch_dict

        gt_boxes = batch_dict['gt_boxes']
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']  # 获取原始点云数据

        # 修复：正确生成batch_mask
        gt_fill_coords_list = []
        for batch_idx in range(batch_size):
            # 创建当前batch的网格坐标
            z_coords = torch.arange(0, self.grid_size[0], device=gt_boxes.device)
            y_coords = torch.arange(0, self.grid_size[1], device=gt_boxes.device)
            x_coords = torch.arange(0, self.grid_size[2], device=gt_boxes.device)

            zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            all_voxel_coords = torch.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=1)

            # 计算体素中心点坐标
            voxel_centers = (all_voxel_coords.float() + 0.5) * self.voxel_size + self.point_cloud_range[:3]

            # 修复：正确获取当前batch的gt_boxes
            batch_gt_boxes = gt_boxes[batch_idx]  # 形状为[max_gt_boxes, 8]

            # 过滤掉无效的gt_boxes（通常用0填充的无效框）
            valid_mask = (batch_gt_boxes[:, 3:6].sum(dim=1) > 0)  # 检查dx,dy,dz是否都为0
            valid_gt_boxes = batch_gt_boxes[valid_mask]

            if len(valid_gt_boxes) == 0:
                continue

            # 检查每个真值框内的体素
            for box_idx, box in enumerate(valid_gt_boxes):
                geometric_box = box[1:8]  # [x, y, z, dx, dy, dz, heading]

                # 修复：将CUDA张量转换为CPU上的numpy数组
                voxel_centers_cpu = voxel_centers.cpu().numpy()
                geometric_box_cpu = geometric_box.unsqueeze(0).cpu().numpy()

                mask = box_utils.points_in_box_3d(voxel_centers_cpu, geometric_box_cpu)
                mask = torch.from_numpy(mask).to(voxel_centers.device)  # 转回CUDA张量

                inside_voxel_coords = all_voxel_coords[mask]

                if len(inside_voxel_coords) > 0:
                    batch_indices = torch.full((len(inside_voxel_coords), 1), batch_idx,
                                               dtype=torch.int32, device=gt_boxes.device)
                    coords_with_batch = torch.cat([batch_indices, inside_voxel_coords], dim=1)
                    gt_fill_coords_list.append(coords_with_batch)

        # 合并所有batch的结果
        if gt_fill_coords_list:
            gt_fill_coords = torch.cat(gt_fill_coords_list, dim=0)
        else:
            gt_fill_coords = torch.zeros((0, 4), dtype=torch.int32, device=gt_boxes.device)

        # 2. 生成真值框参考地图 (gt_reference_map)
        gt_reference_coords_list = []
        if voxel_coords is not None and len(voxel_coords) > 0:
            # 计算体素中心点坐标
            voxel_centers_ref = (voxel_coords[:, 1:4].float() + 0.5) * self.voxel_size + self.point_cloud_range[:3]

            for batch_idx in range(batch_size):
                # 获取当前batch的体素坐标
                batch_mask = voxel_coords[:, 0] == batch_idx
                batch_voxel_coords = voxel_coords[batch_mask]
                batch_voxel_centers = voxel_centers_ref[batch_mask]

                if len(batch_voxel_coords) == 0:
                    continue

                # 修复：正确获取当前batch的gt_boxes
                batch_gt_boxes = gt_boxes[batch_idx]
                valid_mask = (batch_gt_boxes[:, 3:6].sum(dim=1) > 0)
                valid_gt_boxes = batch_gt_boxes[valid_mask]

                if len(valid_gt_boxes) == 0:
                    continue

                # 检查每个真值框内的体素
                for box_idx, box in enumerate(valid_gt_boxes):
                    geometric_box = box[1:8]

                    # 修复：将CUDA张量转换为CPU上的numpy数组
                    batch_voxel_centers_cpu = batch_voxel_centers.cpu().numpy()
                    geometric_box_cpu = geometric_box.unsqueeze(0).cpu().numpy()

                    mask = box_utils.points_in_box_3d(batch_voxel_centers_cpu, geometric_box_cpu)
                    mask = torch.from_numpy(mask).to(batch_voxel_centers.device)  # 转回CUDA张量

                    inside_voxel_coords = batch_voxel_coords[mask]

                    if len(inside_voxel_coords) > 0:
                        gt_reference_coords_list.append(inside_voxel_coords)

        # 合并结果
        if gt_reference_coords_list:
            gt_reference_coords = torch.cat(gt_reference_coords_list, dim=0)
        else:
            gt_reference_coords = torch.zeros((0, 4), dtype=torch.int32, device=gt_boxes.device)

        # 存储到batch_dict
        batch_dict['gt_fill_coords'] = gt_fill_coords
        batch_dict['gt_reference_coords'] = gt_reference_coords


        print(f"=== VFE真值框地图生成结果 ===")
        print(f"真值框填充地图: 找到 {len(gt_fill_coords)} 个体素")
        print(f"真值框参考地图: 找到 {len(gt_reference_coords)} 个体素")


        return batch_dict

    def forward(self, batch_dict, **kwargs):

        # 主流程体素化
        points = batch_dict['points']

        # 计算体素坐标
        points_coords = torch.floor(
            (points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size[[0, 1, 2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()


        # 生成唯一体素坐标
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        # 计算特征
        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # 通过PFN层
        for i, pfn in enumerate(self.pfn_layers):
            features = pfn(features, unq_inv)

        # 生成最终体素坐标
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]  # [batch_idx, x, y, z]

        # 在VFE环节重新生成真值框地图
        batch_dict = self._generate_gt_voxel_maps(batch_dict, voxel_coords, points_xyz)

        # 存储结果 - 确保字段正确设置
        batch_dict['pillar_features'] = features
        batch_dict['voxel_features'] = features  # 两个字段都设置
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict