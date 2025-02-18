from torch import nn
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from data.pelivic_ae import sitk_load, sitk_save

class Sobel3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv3d(in_channels=1, 
                               out_channels=3, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1,  # 添加padding以保持输入输出大小一致
                               bias=False)
        
        # 定义x方向的3D Sobel算子
        # Gx = torch.tensor([
        #     [[2.0, 0.0, -2.0],
        #      [4.0, 0.0, -4.0],
        #      [2.0, 0.0, -2.0]],
            
        #     [[4.0, 0.0, -4.0],
        #      [8.0, 0.0, -8.0],
        #      [4.0, 0.0, -4.0]],
            
        #     [[2.0, 0.0, -2.0],
        #      [4.0, 0.0, -4.0],
        #      [2.0, 0.0, -2.0]]
        # ]) / 128.0
        # Sobel核权重优化（增强骨骼边缘）
        Gx = torch.tensor([
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            [[2, 0, -2], [4, 0, -4], [2, 0, -2]], 
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        ]) / 32.0  # 调整分母系数控制灵敏度
        Gy = Gx.permute(2, 1, 0)
        Gz = Gx.permute(1, 2, 0)
        
        G = torch.stack([Gx, Gy, Gz], dim=0)
        G = G.unsqueeze(1)
        
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        if img.dim() == 4:
            img = img.unsqueeze(0)
        if img.dim() == 3:
            img = img.unsqueeze(0).unsqueeze(0)
            
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

def visualize_3d_edges(volume, edges, slice_indices=None):
    """
    可视化三个正交平面的原始图像和边缘检测结果
    
    参数:
    volume: 原始CT体积数据 [1, 1, D, H, W]
    edges: 边缘检测结果 [1, 1, D, H, W]
    slice_indices: 字典，包含三个方向的切片索引
    """
    if slice_indices is None:
        # 默认使用中间切片
        d, h, w = volume.shape[2:]
        slice_indices = {
            'axial': d // 2,
            'coronal': h // 2,
            'sagittal': w // 2
        }
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle('3D Edge Detection Results', fontsize=16)
    
    # 转换为numpy数组
    volume_np = volume.squeeze().cpu().numpy()
    edges_np = edges.squeeze().cpu().numpy()
    
    # 正则化边缘图像到[0,1]范围
    edges_np = (edges_np - edges_np.min()) / (edges_np.max() - edges_np.min() + 1e-8)
    
    # 轴向切片 (Axial)
    axes[0, 0].imshow(volume_np[slice_indices['axial']], cmap='gray')
    axes[0, 0].set_title('Original (Axial)')
    axes[1, 0].imshow(edges_np[slice_indices['axial']], cmap='jet')
    axes[1, 0].set_title('Edges (Axial)')
    
    # 冠状切片 (Coronal)
    axes[0, 1].imshow(volume_np[:, slice_indices['coronal'], :], cmap='gray')
    axes[0, 1].set_title('Original (Coronal)')
    axes[1, 1].imshow(edges_np[:, slice_indices['coronal'], :], cmap='jet')
    axes[1, 1].set_title('Edges (Coronal)')
    
    # 矢状切片 (Sagittal)
    axes[0, 2].imshow(volume_np[:, :, slice_indices['sagittal']], cmap='gray')
    axes[0, 2].set_title('Original (Sagittal)')
    axes[1, 2].imshow(edges_np[:, :, slice_indices['sagittal']], cmap='jet')
    axes[1, 2].set_title('Edges (Sagittal)')
    
    # 关闭坐标轴
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_3d_surface(edges, threshold=0.5):
    """
    绘制3D边缘的表面图
    
    参数:
    edges: 边缘检测结果
    threshold: 二值化阈值
    """
    edges_np = edges.squeeze().cpu().numpy()
    
    # 标准化到[0,1]
    edges_np = (edges_np - edges_np.min()) / (edges_np.max() - edges_np.min() + 1e-8)
    
    # 二值化
    binary_edges = edges_np > threshold
    
    # 获取非零点的坐标
    z, y, x = np.nonzero(binary_edges)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用scatter绘制点云
    scatter = ax.scatter(x, y, z, c=edges_np[z, y, x], cmap='jet', 
                        alpha=0.1, marker='.', s=1)
    
    plt.colorbar(scatter)
    ax.set_title('3D Edge Surface')
    plt.show()

def process_ct_volume(file_path):
    """
    处理CT体积数据并可视化结果
    """
    # 加载数据
    volume, spacing = sitk_load(file_path)
    
    # 转换为PyTorch tensor
    torch_volume = torch.from_numpy(volume).float()
    torch_volume = torch_volume.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    
    # 标准化到[0,1]范围
    torch_volume = (torch_volume - torch_volume.min()) / (torch_volume.max() - torch_volume.min() + 1e-8)
    
    # 初始化Sobel3D
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sobel = Sobel3D().to(device)
    torch_volume = torch_volume.to(device)
    
    # 应用边缘检测
    with torch.no_grad():
        edges = sobel(torch_volume)
    
    # 可视化结果
    print("Visualizing 2D slices...")
    visualize_3d_edges(torch_volume, edges)
    
    print("Visualizing 3D surface...")
    plot_3d_surface(edges, threshold=0.3)
    
    return edges

if __name__ == "__main__":
    # 替换为你的CT文件路径
    ct_file_path = 'F:/Data_Space/Pelvic1K/processed_128x128_s2.0_block_48/images/dataset6_CLINIC_0005_data_post_centralized.nii.gz'
    edges = process_ct_volume(ct_file_path)