import os
import numpy as np
import torch 
import torch.nn as nn

from ldm.models.implict_autoencoder.util import make_coord ,make_coord_cell , to_pixel_samples

import pdb


if __name__ == '__main__':
    img_tensor = torch.randn(1,3,255,256)
    pdb.set_trace
    #feat_coord = make_coord(shape=shape , ranges=None ,flatten=False)
    coord , cell , gt = to_pixel_samples(img_tensor)
    pdb.set_trace()
    
    