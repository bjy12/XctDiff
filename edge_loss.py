"""
Created on Tue Jul 10 2022
Last Modified on Thu Apr 6 2023

@author: Agamdeep Chopra, achopra4@uw.edu
@affiliation: University of Washington, Seattle WA
@reference: Thevenot, A. (2022, February 17). Implement canny edge detection
            from scratch with PyTorch. Medium. Retrieved July 10, 2022, from
            https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
"""
import torch
import numpy as np
from numpy import asarray, float32
from torch import from_numpy, sum as tsum, stack, cat, float32 as tfloat32
from torch.nn import Module, Conv3d, L1Loss
from torch.nn.functional import pad as tpad


from data.pelivic_ae import sitk_load, sitk_save
import matplotlib.pyplot as plt
import pdb
def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    '''
    Returns 3D Sobel kernels Sx, Sy, Sz, & diagonal kernels for edge detection.

    Parameters
    ----------
    n1 : int, optional
        Kernel value 1 (default 1).
    n2 : int, optional
        Kernel value 2 (default 2).
    n3 : int, optional
        Kernel value 3 (default 2).

    Returns
    -------
    list
        List of all the 3D Sobel kernels (Sx, Sy, Sz, diagonal kernels).
    '''
    Sx = asarray(
        [[[-n1, 0, n1],
          [-n2, 0, n2],
          [-n1, 0, n1]],
         [[-n2, 0, n2],
          [-n3*n2, 0, n3*n2],
          [-n2, 0, n2]],
         [[-n1, 0, n1],
          [-n2, 0, n2],
          [-n1, 0, n1]]])

    Sy = asarray(
        [[[-n1, -n2, -n1],
          [0, 0, 0],
          [n1, n2, n1]],
         [[-n2, -n3*n2, -n2],
          [0, 0, 0],
          [n2, n3*n2, n2]],
         [[-n1, -n2, -n1],
          [0, 0, 0],
          [n1, n2, n1]]])

    Sz = asarray(
        [[[-n1, -n2, -n1],
          [-n2, -n3*n2, -n2],
          [-n1, -n2, -n1]],
         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
         [[n1, n2, n1],
          [n2, n3*n2, n2],
          [n1, n2, n1]]])

    Sd11 = asarray(
        [[[0, n1, n2],
          [-n1, 0, n1],
          [-n2, -n1, 0]],
         [[0, n2, n2*n3],
          [-n2, 0, n2],
          [-n2*n3, -n2, 0]],
         [[0, n1, n2],
          [-n1, 0, n1],
          [-n2, -n1, 0]]])

    Sd12 = asarray(
        [[[-n2, -n1, 0],
          [-n1, 0, n1],
          [0, n1, n2]],
         [[-n2*n3, -n2, 0],
          [-n2, 0, n2],
          [0, n2, n2*n3]],
         [[-n2, -n1, 0],
          [-n1, 0, n1],
          [0, n1, n2]]])

    Sd21 = Sd11.T
    Sd22 = Sd12.T
    Sd31 = asarray([-S.T for S in Sd11.T])
    Sd32 = asarray([S.T for S in Sd12.T])

    return [Sx, Sy, Sz, Sd11, Sd12, Sd21, Sd22, Sd31, Sd32]


class GradEdge3D():
    '''
    Implements Sobel edge detection for 3D images using PyTorch.

    Parameters
    ----------
    n1 : int, optional
        Filter size for the first dimension (default is 1).
    n2 : int, optional
        Filter size for the second dimension (default is 2).
    n3 : int, optional
        Filter size for the third dimension (default is 2).
    '''

    def __init__(self, n1=1, n2=2, n3=2):
        super(GradEdge3D, self).__init__()
        k_sobel = 3
        S = get_sobel_kernel3D(n1, n2, n3)
        self.sobel_filters = []

        # Initialize Sobel filters for edge detection
        for s in S:
            sobel_filter = Conv3d(
                in_channels=1, out_channels=1, stride=1,
                kernel_size=k_sobel, padding=k_sobel // 2, bias=False)
            sobel_filter.weight.data = from_numpy(
                s.astype(float32)).reshape(1, 1, k_sobel, k_sobel, k_sobel)
            sobel_filter = sobel_filter.to(dtype=tfloat32)
            self.sobel_filters.append(sobel_filter)

    def __call__(self, img, a=1):
        '''
        Perform edge detection on the given 3D image.

        Parameters
        ----------
        img : torch.Tensor
            3D input tensor of shape (B, C, x, y, z).
        a : int, optional
            Padding size (default is 1).

        Returns
        -------
        torch.Tensor
            Tensor containing the gradient magnitudes of the edges.
        '''
        pad = (a, a, a, a, a, a)
        B, C, H, W, D = img.shape

        img = tpad(img, pad, mode='reflect')

        # Calculate gradient magnitude of edges
        grad_mag = (1 / C) * tsum(stack([tsum(cat(
            [s.to(img.device)(img[:, c:c+1]) for c in range(C)],
            dim=1) + 1e-6, dim=1) ** 2 for s in self.sobel_filters],
            dim=1) + 1e-6, dim=1) ** 0.5
        grad_mag = grad_mag[:, a:-a, a:-a, a:-a]

        num = grad_mag - grad_mag.min()
        dnm = grad_mag.max() - grad_mag.min() + 1e-6
        norm_grad_mag = num / dnm

        return norm_grad_mag.view(B, 1, H, W, D)


class GMELoss3D(Module):
    '''
    Implements Gradient Magnitude Edge Loss for 3D image data.

    Parameters
    ----------
    n1 : int
        Filter size for the first dimension.
    n2 : int
        Filter size for the second dimension.
    n3 : int
        Filter size for the third dimension.
    lam_errors : list
        List of tuples (weight, loss function) for computing error.
    reduction : str
        Reduction method for loss ('sum' or 'mean').
    '''

    def __init__(self, n1=1, n2=2, n3=2,
                 lam_errors=[(1.0, L1Loss())], reduction='sum'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(n1, n2, n3)
        self.lam_errors = lam_errors
        self.reduction = reduction

    def forward(self, x, y):
        '''
        Compute the loss based on the edges detected in the input tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, x, y, z).
        y : torch.Tensor
            Target tensor of shape (B, C, x, y, z).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        '''
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'

        edge_x = self.edge_filter(x)
        edge_y = self.edge_filter(y)

        if self.reduction == 'sum':
            error = 1e-6 + sum([lam * err_func(edge_x, edge_y)
                                for lam, err_func in self.lam_errors])
        else:
            error = 1e-6 + (
                sum(
                    [lam * err_func(
                        edge_x, edge_y) for lam, err_func in self.lam_errors]
                ) / len(self.lam_errors))

        return error
    
def load_nii_to_tensor(path, normalize=True):
    """加载nii.gz文件并转换为PyTorch张量"""
    # 加载nii文件
    np_img , spacing  =  sitk_load(path)
    
    # 转换为PyTorch张量并添加批次和通道维度
    tensor = torch.from_numpy(np_img.astype(np.float32))
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    # 归一化到[-1,1]
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 2 - 1
        
    return tensor, spacing

def visualize_gradients(original, gradients, slice_idx=None):
    """可视化原始图像和梯度图的中间切片"""
    # 转换为numpy数组
    original_np = original.squeeze().cpu().numpy()
    grad_np = gradients.squeeze().cpu().numpy()
    
    # 自动选择中间切片
    if slice_idx is None:
        slice_idx = [s//2 for s in grad_np.shape]
    
    # 创建可视化窗口
    plt.figure(figsize=(15, 10))
    
    # 原始图像切片
    plt.subplot(231, title="Original Axial")
    plt.imshow(original_np[slice_idx[0], :, :], cmap='gray', clim=(-1, 1))
    plt.axis('off')
    
    plt.subplot(232, title="Original Coronal")
    plt.imshow(original_np[:, slice_idx[1], :], cmap='gray', clim=(-1, 1))
    plt.axis('off')
    
    plt.subplot(233, title="Original Sagittal")
    plt.imshow(original_np[:, :, slice_idx[2]], cmap='gray', clim=(-1, 1))
    plt.axis('off')
    
    # 梯度图切片
    plt.subplot(234, title="Gradient Axial")
    plt.imshow(grad_np[slice_idx[0], :, :], cmap='jet', clim=(0, 1))
    plt.axis('off')
    
    plt.subplot(235, title="Gradient Coronal")
    plt.imshow(grad_np[:, slice_idx[1], :], cmap='jet', clim=(0, 1))
    plt.axis('off')
    
    plt.subplot(236, title="Gradient Sagittal")
    plt.imshow(grad_np[:, :, slice_idx[2]], cmap='jet', clim=(0, 1))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_gradient_nii(grad_tensor, spacing, output_path):
    """保存梯度图为nii.gz文件"""
    # 转换为numpy数组并调整轴顺序
    grad_np = grad_tensor.squeeze().cpu().numpy()
    sitk_save(output_path ,grad_np ,spacing=spacing , uint8=False)
    # 创建Nifti图像对象
    # 保存文件
    print(f"Gradient map saved to {output_path}")

if __name__ == "__main__":
    example_path = 'F:/Data_Space/Pelvic1K/processed_128x128_s2.0_block_48/images/dataset6_CLINIC_0005_data_post_centralized.nii.gz'
    img_tensor , spacing = load_nii_to_tensor(example_path)

    edge_detector = GradEdge3D(n1=1,n2=1,n3=1)

    with torch.no_grad():
        grad_map = edge_detector(img_tensor)
    pdb.set_trace()
    visualize_gradients(img_tensor, grad_map)

    save_gradient_nii(grad_map , spacing=spacing , output_path='grad_map.nii.gz')