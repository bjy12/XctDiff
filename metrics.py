import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pdb

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # Ensure the images are in the correct range [0, 1]
    img1 = (img1 + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
    img2 = (img2 + 1) / 2.0
    
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Ensure the images are in the correct range [0, 1]
    img1 = (img1 + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
    img2 = (img2 + 1) / 2.0
    
    return ssim(img1, img2, data_range=1.0)

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error between two images"""
    return np.mean(np.abs(img1 - img2))