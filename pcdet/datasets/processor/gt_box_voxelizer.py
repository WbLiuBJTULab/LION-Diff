import numpy as np
from ...utils import box_utils, common_utils


# 在GTBoxVoxelGenerator中引入VoxelGeneratorWrapper
class GTBoxVoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, num_point_features, max_points_per_voxel, max_voxels):
        self.voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features,
            max_num_points_per_voxel=max_points_per_voxel,
            max_num_voxels=max_voxels
        )
        self.grid_size = self.voxel_generator.grid_size  # 从wrapper获取一致网格尺寸

    def generate(self, points, gt_boxes):
        # 使用DataProcessor的体素生成器
        _, all_coordinates, _ = self.voxel_generator.generate(points)
        gt_voxel_map = np.zeros(self.grid_size[::-1], dtype=np.float32)

        # 仅标记真值框内的体素坐标
        for box in gt_boxes:
            corners = box_utils.boxes_to_corners_3d(box.reshape(1, 7))[0]

            # 改为 (兼容不同spconv版本)：
            if all_coordinates.shape[1] == 4:  # spconv v1格式 [batch_idx, z, y, x]
                coords = all_coordinates[:, 1:4]
            else:  # spconv v2格式 [z, y, x]
                coords = all_coordinates[:, :3]
            mask = box_utils.in_hull(coords, corners)

            for coord in all_coordinates[mask]:
                z, y, x = coord
                gt_voxel_map[z, y, x] = 1.0
        return gt_voxel_map