import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data.pelivic_ae import sitk_load, sitk_save
import pdb


class BoneEdgeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv3d(in_channels=1, 
                               out_channels=3, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1,
                               bias=False)
        
        # 调整Sobel算子
        Gx = torch.tensor([
            [[3.0, 0.0, -3.0],
             [6.0, 0.0, -6.0],
             [3.0, 0.0, -3.0]],
            
            [[6.0, 0.0, -6.0],
             [12.0, 0.0, -12.0],
             [6.0, 0.0, -6.0]],
            
            [[3.0, 0.0, -3.0],
             [6.0, 0.0, -6.0],
             [3.0, 0.0, -3.0]]
        ]) / 192.0

        Gy = Gx.permute(2, 1, 0)
        Gz = Gx.permute(1, 2, 0)
        
        G = torch.stack([Gx, Gy, Gz], dim=0)
        G = G.unsqueeze(1)
        
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, x):
        # 标准化到[0,1]范围
        x = x / 255.0
        
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4:
            x = x.unsqueeze(0)
            
        gradients = self.filter(x)
        pdb.set_trace()
        gradient_magnitude = torch.sqrt(torch.sum(gradients**2, dim=1, keepdim=True))

        # 将结果缩放回[0,255]范围
        return gradient_magnitude * 255.0

def visualize_bone_edges(volume, edges, slice_indices=None):
    """可视化骨骼边缘检测结果"""
    if slice_indices is None:
        d, h, w = volume.shape[2:]
        slice_indices = {
            'axial': d // 2,
            'coronal': h // 2,
            'sagittal': w // 2
        }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle('Bone Edge Detection Results', fontsize=16)
    
    volume_np = volume.squeeze().cpu().numpy()
    edges_np = edges.squeeze().cpu().numpy()
    
    # 轴向切片
    axes[0, 0].imshow(volume_np[slice_indices['axial']], cmap='gray')
    axes[0, 0].set_title('Original (Axial)')
    edges_ax = axes[1, 0].imshow(edges_np[slice_indices['axial']], cmap='hot')
    axes[1, 0].set_title('Bone Edges (Axial)')
    
    # 冠状切片
    axes[0, 1].imshow(volume_np[:, slice_indices['coronal'], :], cmap='gray')
    axes[0, 1].set_title('Original (Coronal)')
    edges_cor = axes[1, 1].imshow(edges_np[:, slice_indices['coronal'], :], cmap='hot')
    axes[1, 1].set_title('Bone Edges (Coronal)')
    
    # 矢状切片
    axes[0, 2].imshow(volume_np[:, :, slice_indices['sagittal']], cmap='gray')
    axes[0, 2].set_title('Original (Sagittal)')
    edges_sag = axes[1, 2].imshow(edges_np[:, :, slice_indices['sagittal']], cmap='hot')
    axes[1, 2].set_title('Bone Edges (Sagittal)')
    
    # 添加颜色条
    fig.colorbar(edges_ax, ax=axes[1, 0], label='Edge Strength')
    fig.colorbar(edges_cor, ax=axes[1, 1], label='Edge Strength')
    fig.colorbar(edges_sag, ax=axes[1, 2], label='Edge Strength')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_hip_ct(file_path):
    """处理髋部CT图像"""
    # 加载数据
    volume, spacing = sitk_load(file_path)
    
    # 转换为PyTorch tensor
    torch_volume = torch.from_numpy(volume).float()
    torch_volume = torch_volume.unsqueeze(0).unsqueeze(0)
    
    # 初始化检测器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = BoneEdgeDetector().to(device)
    torch_volume = torch_volume.to(device)
    
    # 检测边缘
    with torch.no_grad():
        edges = detector(torch_volume)
    
    # 可视化
    visualize_bone_edges(torch_volume, edges)
    
    return edges, torch_volume

if __name__ == "__main__":
    ct_file_path = 'F:/Data_Space/Pelvic1K/processed_128x128_s2.0_block_48/images/dataset6_CLINIC_0005_data_post_centralized.nii.gz'
    edges, volume = process_hip_ct(ct_file_path)
    print("Input range:", volume.min().item(), "to", volume.max().item())
    print("Output range:", edges.min().item(), "to", edges.max().item())