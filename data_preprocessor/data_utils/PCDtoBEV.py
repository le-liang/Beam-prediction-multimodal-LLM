import torch
import numpy as np
import yaml
import open3d as o3d
from easydict import EasyDict as edict
import re
from cumm import tensorview as tv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # 新增：用于绘图

from opencood.models.point_pillar_where2comm import PointPillarWhere2comm
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


def model_loader(cfg_file, ckpt_file, device):
    stream = open(cfg_file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    config = edict(param)

    model = PointPillarWhere2comm(config.model.args)

    checkpoint = torch.load(ckpt_file, map_location=device)
    full_state_dict = checkpoint.get('model_state_dict', checkpoint)

    point_pillar_state_dict = {}
    for k, v in full_state_dict.items():
        if k in model.state_dict():
            point_pillar_state_dict[k] = v
        elif k.startswith('encoder.'):
            new_key = k.replace('encoder.', '', 1)
            if new_key in model.state_dict():
                point_pillar_state_dict[new_key] = v

    model.load_state_dict(point_pillar_state_dict, strict=True)

    model.to(device)
    model.eval()
    return model, config


def PCDtoBEV(model, config, pcd_file, device, visualize=True, save_path=None):  # 新增 visualize 参数
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)

    # 修正坐标系：交换 X 和 Y 轴
    # 应对 LiDAR 多转了 90 度的情况，将数据变换为 X(前后)短，Y(左右)长
    xyz[:, [0, 1]] = xyz[:, [1, 0]]

    # we save the intensity in the first channel
    if pcd.has_colors():
        intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    else:
        # Fallback if no colors/intensity found
        intensity = np.zeros((xyz.shape[0], 1))

    pcd_np = np.hstack((xyz, intensity))

    pcd_np = np.asarray(pcd_np, dtype=np.float32)
    spcon = 1
    try:
        # spconv v1.x
        from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
    except:
        # spconv v2.x
        from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

        spcon = 2

    if spcon == 1:
        voxel_generator = VoxelGenerator(
            voxel_size=config.preprocess.args.voxel_size,
            point_cloud_range=config.model.args.lidar_range,
            max_num_points=config.preprocess.args.max_points_per_voxel,
            max_voxels=config.preprocess.args.max_voxel_train
        )
    else:
        voxel_generator = VoxelGenerator(
            vsize_xyz=config.preprocess.args.voxel_size,
            coors_range_xyz=config.model.args.lidar_range,
            max_num_points_per_voxel=config.preprocess.args.max_points_per_voxel,
            num_point_features=4,
            max_num_voxels=config.preprocess.args.max_voxel_train
        )

    data_dict = {}
    if spcon == 1:
        voxel_output = voxel_generator.generate(pcd_np)
    else:
        pcd_tv = tv.from_numpy(pcd_np)
        voxel_output = voxel_generator.point_to_voxel(pcd_tv)
    if isinstance(voxel_output, dict):
        voxels, coordinates, num_points = \
            voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
    else:
        voxels, coordinates, num_points = voxel_output

    if spcon == 2:
        voxels = voxels.numpy()
        coordinates = coordinates.numpy()
        num_points = num_points.numpy()

    voxels = torch.from_numpy(voxels).to(device).float()

    batch_idx_col = np.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)
    voxel_coords = np.hstack((batch_idx_col, coordinates))
    voxel_coords = torch.from_numpy(voxel_coords).to(device).float()

    num_points = torch.from_numpy(num_points).to(device).float()

    data_dict['voxel_features'] = voxels
    data_dict['voxel_num_points'] = num_points
    data_dict['voxel_coords'] = voxel_coords

    # 3. bev_feature_extraction
    # 使用 no_grad 避免梯度计算，节省显存
    with torch.no_grad():
        batch_dict = model.pillar_vfe(data_dict)
        batch_dict = model.scatter(batch_dict)
        batch_dict = model.backbone(batch_dict)

        # spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']

        spatial_features_2d = model.shrink_conv(spatial_features_2d)
        spatial_features_2d = spatial_features_2d.squeeze(0)

    # --- 可视化逻辑 ---
    if visualize:
        # 1. 转为 Numpy (C, H, W)
        feature_map = spatial_features_2d.cpu().numpy()

        # 2. 通道压缩：取最大值 (Max Projection)
        # 物理意义：只要某个位置在任意特征通道上有高响应，就认为该位置有物体
        heatmap = np.max(feature_map, axis=0)

        # 3. 绘图
        plt.figure(figsize=(10, 6))
        # origin='lower' 非常重要，因为 LiDAR 坐标系通常 Y 轴朝前/上
        plt.imshow(heatmap, cmap='jet', origin='lower')
        plt.colorbar(label='Feature Activation')

        H, W = heatmap.shape
        plt.title(f"BEV Feature Map Visualization\nResolution: {H}x{W}")
        plt.xlabel("X (Width)")
        plt.ylabel("Y (Height)")

        # 标出 RSU（BEV 网格的中心）
        center_x = W / 2.0
        center_y = H / 2.0
        # 用红色叉号标出并添加文本标签
        plt.scatter([center_x], [center_y], c='red', s=60, marker='x')
        plt.text(center_x + 2, center_y + 2, 'RSU', color='red', fontsize=10)

        # 保存或显示图片（当 save_path 为 None 时调用 plt.show）
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()
    # -----------------

    return spatial_features_2d
