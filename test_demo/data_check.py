import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import pickle
import yaml 
import pdb
def visualize_multiple_projections(projs, angles, num_images=9):
    # 选择要显示的投影索引
    indices = np.linspace(0, len(projs)-1, num_images, dtype=int)
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # 创建图形
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle('Multiple Projections View', fontsize=16)
    
    # 展示图像
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            proj_idx = indices[idx]
            ax.imshow(projs[proj_idx], cmap='gray')
            ax.set_title(f'Angle: {angles[proj_idx]:.1f}°')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
# 使用函数
pickle_path  = 'F:/Data_Space/Pelvic1K/processed_128x128_s2.0_block_48/projections/dataset6_CLINIC_0014_data_post_centralized.pickle'
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
    projs = data['projs']         # uint8: [K, W, H]
    projs_max = data['projs_max'] # float
    angles = data['angles']       # float: [K,]

visualize_multiple_projections(projs , angles , num_images=2)