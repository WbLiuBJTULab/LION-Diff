from functools import partial

import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):

        # 使用spconv库生成体素
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask_z = config.get('MASK_Z', False)
            if mask_z:
                mask = common_utils.mask_points_by_range_v2(data_dict['points'], self.point_cloud_range)
            else:
                mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        # 初始化体素生成器

        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE, # 体素尺寸
                coors_range_xyz=self.point_cloud_range, # 点云范围
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL, # 单个体素最大点数
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        # 生成体素
        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        # 根据标志移除XYZ坐标
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        # 存入data_dict
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        return data_dict

    # 20250905 _代码修改 直接在data_processor中加入真值框地图的功能
    # 20251010 _代码修改 修改以创建两个地图功能
    # === 新增方法：真值框体素地图生成 ===
    def generate_gt_voxel_maps(self, data_dict=None, config=None):
        """
        生成两种真值框地图（兼容placeholder模式）：
        1. gt_fill_coords: 基于网格坐标系的所有真值框体素坐标
        2. gt_reference_coords: 基于点云体素化的真值框内体素坐标
        """

        if data_dict is None:
            # 初始化阶段设置网格尺寸（与placeholder模式一致）
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.generate_gt_voxel_maps, config=config)

        # 仅在训练模式且有真值框时执行
        if self.training and 'gt_boxes' in data_dict and len(data_dict['gt_boxes']) > 0:

            # 1. 生成真值框填充地图（这部分与placeholder模式兼容）
            # 创建网格坐标系中的所有体素坐标
            z_coords = np.arange(0, self.grid_size[0])
            y_coords = np.arange(0, self.grid_size[1])
            x_coords = np.arange(0, self.grid_size[2])
            zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            all_voxel_coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

            # 将体素坐标转换为实际点云坐标（中心点）
            voxel_centers = (all_voxel_coords + 0.5) * np.array(self.voxel_size) + np.array(self.point_cloud_range[:3])

            gt_fill_coords = []
            for box in data_dict['gt_boxes']:
                geometric_box = box[:7]  # [x,y,z,dx,dy,dz,heading]
                mask = box_utils.points_in_box_3d(voxel_centers, geometric_box)
                inside_voxel_coords = all_voxel_coords[mask]
                gt_fill_coords.extend(inside_voxel_coords)

            # 2. 生成真值框参考地图（需要处理placeholder模式）
            gt_reference_coords = []

            # 检查是否已经存在体素坐标（KITTI模式）
            if 'voxel_coords' in data_dict and data_dict['voxel_coords'] is not None:
                # KITTI模式：使用已有的体素坐标
                all_coordinates = data_dict['voxel_coords']
            else:
                # placeholder模式：临时生成体素坐标用于GT地图
                if self.voxel_generator is None:
                    self.voxel_generator = VoxelGeneratorWrapper(
                        vsize_xyz=config.VOXEL_SIZE,
                        coors_range_xyz=self.point_cloud_range,
                        num_point_features=self.num_point_features,
                        max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                        max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                    )

                points = data_dict['points']
                _, all_coordinates, _ = self.voxel_generator.generate(points)

            # 处理坐标格式兼容性
            if all_coordinates.shape[1] == 4:  # spconv v1: [batch_idx, z, y, x]
                coords = all_coordinates[:, 1:4]
            else:  # spconv v2: [z, y, x]
                coords = all_coordinates[:, :3]

            voxel_centers_ref = (coords + 0.5) * np.array(self.voxel_size) + np.array(self.point_cloud_range[:3])

            for box in data_dict['gt_boxes']:
                geometric_box = box[:7]
                mask = box_utils.points_in_box_3d(voxel_centers_ref, geometric_box)
                inside_voxel_coords = coords[mask]
                gt_reference_coords.extend(inside_voxel_coords)

            # 存储结果
            data_dict['gt_fill_coords'] = np.array(gt_fill_coords, dtype=np.int32) \
                if gt_fill_coords else np.zeros((0, 3), dtype=np.int32)
            data_dict['gt_reference_coords'] = np.array(gt_reference_coords, dtype=np.int32) \
                if gt_reference_coords else np.zeros((0, 3), dtype=np.int32)

        return data_dict
    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
