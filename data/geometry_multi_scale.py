import torch 
import torch.nn as nn
import numpy as np
import os
from copy import deepcopy
import yaml
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Multi_Scale_Geometry(object):
    def __init__(self, config , v_scale_factor , p_scale_factor):
        self.v_res = config['nVoxel'][0]    # ct scan
        self.p_res = config['nDetector'][0] # projections
        self.v_spacing = np.array(config['dVoxel'])[0]    # mm
        self.p_spacing = np.array(config['dDetector'])[0] # mm
        # NOTE: only (res * spacing) is used
        #pdb.set_trace()
        self.multi_v_res  = self.v_res / v_scale_factor # []
        self.multi_v_spacing = self.v_spacing * v_scale_factor

        self.multi_p_res = self.p_res / p_scale_factor
        self.multi_p_spacing = self.p_spacing * p_scale_factor 

        self.DSO = config['DSO'] # mm, source to origin
        self.DSD = config['DSD'] # mm, source to detector
    def get_multi_v_res(self):
        multi_v_res = self.multi_v_res
        return multi_v_res
    def project_multi_scale_points(self, points , angle , scale_level):
        #pdb.set_trace()
        d1 = self.DSO
        d2 = self.DSD

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        #pdb.set_trace()
        points *= self.multi_v_res[scale_level] * self.multi_v_spacing[scale_level] # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T
        
        coeff = (d2) / (d1 - points[:, 0]) # N,
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        #pdb.set_trace()
        d_points /= (self.multi_p_res[scale_level] * self.multi_p_spacing[scale_level])
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]
        return d_points


    def project(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        d1 = self.DSO
        d2 = self.DSD

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        points = points @ rot_M.T
        
        coeff = (d2) / (d1 - points[:, 0]) # N,
        #pdb.set_trace()
        d_points = points[:, [2, 1]] * coeff[:, None] # [N, 2] float
        d_points /= (self.p_res * self.p_spacing)
        d_points *= 2 # NOTE: some points may fall outside [-1, 1]
        return d_points
    
    def calculate_projection_distance(self ,  points , angle ):
        d1 = self.DSO
        d2 = self.DSD 

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5 # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2] # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing # mm

        angle = -1 * angle # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1]
        ])
        rotated_points = points @ rot_M.T  
        source_to_point = d1 - rotated_points[:, 0]  # 源点到物体点的距离
        distances = np.abs(d2 - source_to_point)  # 点到探测器平面的距离        return coeff 
        distances_ratio = distances / d2
     
        #pdb.set_trace() 
        return distances_ratio
    
def create_multi_res_points(multi_scale_low_res):
    # multi_scale_res is numpy array 
    multi_scale_low_points = []
    for i in range(len(multi_scale_low_res)):
        low_res_points = create_low_res_points(multi_scale_low_res[i]) 
        multi_scale_low_points.append(low_res_points)
    
    return multi_scale_low_points
def create_low_res_points(low_res):
    resolution = np.array([low_res, low_res, low_res], dtype=np.int32)      
    x, y, z = np.meshgrid(
        np.arange(resolution[0]),
        np.arange(resolution[1]),
        np.arange(resolution[2]),
        indexing='ij'
    )
    #pdb.set_trace()
    coords = np.stack([
        x.flatten(),
        y.flatten(),
        z.flatten()
    ], axis=1)

    # 归一化到 [0,1] 范围
    coords = coords / (np.array(resolution) - 1)
    #pdb.set_trace()
    # 添加batch维度，转换为float32类型
    coords = coords.reshape(-1, 3).astype(np.float32)

    return coords

def project_multi_res_points(multi_res_points , angles ,geo):
    #pdb.set_trace()
    multi_scale_project_points = {}
    for i in range(len(multi_res_points)):
        points_proj = []
        for a in angles:
            p = geo.project_multi_scale_points(multi_res_points[i] , a , scale_level = i )
            points_proj.append(p)
        points_proj = np.stack(points_proj , axis=0)
        #pdb.set_trace()
        multi_scale_project_points[f'level_{i}'] = points_proj
    #db.set_trace()
    return multi_scale_project_points
def visualize_3d_points(low_res_points, title_prefix="Original"):
    """
    可视化3D空间中的点云
    """
    n_scales = len(low_res_points)
    fig = plt.figure(figsize=(5*n_scales, 4))
    
    for i in range(n_scales):
        points = low_res_points[i]
        ax = fig.add_subplot(1, n_scales, i+1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title_prefix} Points - Scale Level {i}')
    
    plt.tight_layout()
    plt.show()

def visualize_projected_points(proj_points, angles):
    """
    可视化投影后的2D点云
    """
    n_angles = len(angles)
    n_scales = len(proj_points)
    
    fig, axes = plt.subplots(n_scales, n_angles, figsize=(5*n_angles, 4*n_scales))
    if n_scales == 1:
        axes = axes[None, :]
    
    for i in range(n_scales):
        points = proj_points[f'level_{i}']
        for j, angle in enumerate(angles):
            ax = axes[i, j]
            ax.scatter(points[j, :, 0], points[j, :, 1], c='r', marker='.', alpha=0.5)
            ax.set_xlabel('X projected')
            ax.set_ylabel('Y projected')
            ax.set_title(f'Projected Points - Scale {i}, Angle {angle:.2f}')
            ax.grid(True)
            # 设置合适的显示范围，考虑到投影可能超出[-1,1]
            max_range = max(abs(points[j].min()), abs(points[j].max())) * 1.1
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            # 添加参考线
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_test_feature_maps(batch_size, channels, heights, widths):
    """
    创建测试用的特征图，每个尺度一个
    使用不同的pattern以便清晰地看出采样位置
    """
    feature_maps = []
    for h, w in zip(heights, widths):
        # 创建棋盘格pattern
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        pattern1 = torch.sin(xx * np.pi * 4) * torch.sin(yy * np.pi * 4)
        pattern2 = torch.cos(xx * np.pi * 2) * torch.cos(yy * np.pi * 2)
        
        feat_map = torch.stack([pattern1, pattern2], dim=0)  # [2, H, W]
        feat_map = feat_map.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 2, H, W]
        feature_maps.append(feat_map)
    
    return feature_maps

def visualize_sampling(feature_maps, proj_points, angles, spacings, detector_spacings):
    """
    可视化在不同尺度下的特征采样效果，考虑spacing信息
    Args:
        feature_maps: 列表，包含不同尺度的特征图
        proj_points: 投影点字典
        angles: 投影角度数组
        spacings: 列表，包含不同尺度的体素spacing [多个(dx,dy,dz)]
        detector_spacings: 列表，包含不同尺度的探测器spacing [多个(du,dv)]
    """
    n_scales = len(feature_maps)
    n_angles = len(angles)
    
    fig, axes = plt.subplots(n_scales, n_angles * 2, figsize=(5*n_angles*2, 4*n_scales))
    if n_scales == 1:
        axes = axes[None, :]
    
    for scale_idx in range(n_scales):
        feat_map = feature_maps[scale_idx]
        points = proj_points[f'level_{scale_idx}']  # [n_angles, N, 2]
        spacing = spacings[scale_idx]
        det_spacing = detector_spacings[scale_idx]
        
        for angle_idx, angle in enumerate(angles):
            # 获取当前角度的投影点
            curr_points = torch.from_numpy(points[angle_idx]).float().unsqueeze(0)  # [1, N, 2]
            
            # 进行特征采样
            sampled_features = index_2d(feat_map, curr_points)  # [1, C, N]
            
            # 可视化原始特征图和采样点
            ax_feat = axes[scale_idx, angle_idx * 2]
            
            # 计算物理尺寸的范围
            h, w = feat_map.shape[2:]
            extent_x = [-w/2 * spacing[0], w/2 * spacing[0]]
            extent_y = [-h/2 * spacing[1], h/2 * spacing[1]]
            
            im = ax_feat.imshow(feat_map[0, 0].numpy(), 
                              extent=[extent_x[0], extent_x[1], extent_y[0], extent_y[1]], 
                              cmap='viridis')
            
            # 转换投影点到物理坐标
            physical_points_x = points[angle_idx, :, 0] * det_spacing[0] * (w/2)
            physical_points_y = points[angle_idx, :, 1] * det_spacing[1] * (h/2)
            
            ax_feat.scatter(physical_points_x, physical_points_y, 
                          c='r', marker='.', alpha=0.5, s=1)
            
            ax_feat.set_title(f'Scale {scale_idx}, Angle {angle:.2f}\n' + 
                            f'Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f})')
            ax_feat.set_xlabel('X (mm)')
            ax_feat.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax_feat)
            
            # # 可视化采样结果
            # ax_samp = axes[scale_idx, angle_idx * 2 + 1]
            # grid_size = int(np.sqrt(points.shape[1]))
            # pdb.set_trace()
            # sampled_grid = sampled_features[0, 0].reshape(grid_size, grid_size).numpy()
            
            # # 计算采样结果的物理尺寸
            # extent_samp_x = [-grid_size/2 * det_spacing[0], grid_size/2 * det_spacing[0]]
            # extent_samp_y = [-grid_size/2 * det_spacing[1], grid_size/2 * det_spacing[1]]
            
            # im = ax_samp.imshow(sampled_grid, 
            #                    extent=[extent_samp_x[0], extent_samp_x[1], 
            #                          extent_samp_y[0], extent_samp_y[1]], 
            #                    cmap='viridis')
            # ax_samp.set_title(f'Scale {scale_idx}, Angle {angle:.2f}\n' + 
            #                 f'Det Spacing: ({det_spacing[0]:.2f}, {det_spacing[1]:.2f})')
            # ax_samp.set_xlabel('X (mm)')
            # ax_samp.set_ylabel('Y (mm)')
            # plt.colorbar(im, ax=ax_samp)
    
    plt.tight_layout()
    plt.show()
def index_2d(feat, uv):
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    feat = feat.transpose(2, 3)  # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]
if __name__ == '__main__':

    config_path = './geo_config/config_2d_256_s2.0_3d_128_s2.5.yaml'
    v_scale_factor = np.array([4,8,16])
    p_sacle_factor = np.array([1,4,8])
    with open(os.path.join(config_path) , 'r') as f:
        geo_config = yaml.safe_load(f)
        geo_ = Multi_Scale_Geometry(geo_config['projector'] , v_scale_factor , p_sacle_factor)
    multi_scale_low_res = geo_.get_multi_v_res()
    pdb.set_trace()
    low_res_points = create_multi_res_points(multi_scale_low_res)
    pdb.set_trace()
    angles = np.array([0.        , 1.57079633])
    proj_multi_scale_points = project_multi_res_points(low_res_points , angles=angles, geo=geo_)
    pdb.set_trace()

    #visualize_3d_points(low_res_points)

    #visualize_projected_points(proj_multi_scale_points ,  angles)



    # 特征图参数
    batch_size = 1
    channels = 2
    heights = [256, 256, 256]  # 根据你的实际尺度调整
    widths = heights

    # 创建测试特征图
    feature_maps = create_test_feature_maps(batch_size, channels, heights, widths)
    # 需要你提供的spacing信息
    pdb.set_trace()

    v_volume_s =  geo_.multi_v_spacing
    p_spacing_ = geo_.multi_p_spacing
    pdb.set_trace()
    volume_spacings = [
        (10.0, 10.0, 10.0),  # 第一个尺度的体素间距
        (20.0, 20.0, 20.0),  # 第二个尺度的体素间距
        (40.0, 40.0, 40.0)   # 第三个尺度的体素间距
    ]

    detector_spacings = [
        (2.0, 2.0),  # 第一个尺度的探测器间距
        (8.0, 8.0),  # 第二个尺度的探测器间距
        (16.0, 16.0)   # 第三个尺度的探测器间距
    ]

    visualize_sampling(feature_maps, proj_multi_scale_points, angles, volume_spacings, detector_spacings)
