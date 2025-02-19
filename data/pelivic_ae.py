import torch
import numpy as np
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from data.geometry import Geometry
import SimpleITK as sitk
import pickle
import os
import yaml
import pdb
PATH_DICT = {
'image': 'images/{}.nii.gz',
'projs': 'projections/{}.pickle',
'projs_vis': 'projections_vis/{}.png',
'blocks_vals': 'blocks/{}_block-{}.npy',
'blocks_coords': 'blocks/blocks_coords.npy'}

def sitk_load(path, uint8=False, spacing_unit='mm'):
    # load as float32
    itk_img = sitk.ReadImage(path)
    #pdb.set_trace()
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    return image, spacing

def sitk_save(path, image, spacing=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)

class LIDC_IDRI_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self._loaditem(index)

    def _transform(self, im, is_drr=False):
        if is_drr:
            im = im.transpose(1, 0)
            im = im[:, ::-1]
            im = im / 255
        else:
            im = (im - im.min()) / (im.max() - im.min()) * 2 - 1.0
        im = np.expand_dims(im, axis=0)
        return torch.from_numpy(np.ascontiguousarray(im)).to(torch.float32)

    def _loaditem(self, index):
        dict = self.data[index]
        im = nib.load(dict['image'])
        filename = dict['xray'].split('/')[-1].split('.')[0]

        xray = Image.open(dict['xray'])
        xray = xray.convert("L")
        xray = xray.resize((128, 128))
        xray = np.array(xray)
        affine = im.affine
        return {
            "image" : self._transform(im.get_fdata()),
            "xray": self._transform(xray, is_drr=True),
            "affine": torch.from_numpy(affine).to(torch.float32),
            "filename": filename
        }

class Pelivc_LatentDiffusionDataset(Dataset):
    def __init__(self , root_data ,file_names , path_dict , mode='autoencoder' , geo_config_path = None):
        super().__init__()

        self.files_names = file_names
        self.data_root = root_data
        self._path_dict = deepcopy(path_dict)
        for key in self._path_dict.keys():
            path = os.path.join(self.data_root, self._path_dict[key])
            self._path_dict[key] = path
        #pdb.set_trace()

        self.mode = mode
        if self.mode == 'ldm':
            with open(os.path.join(geo_config_path) , 'r') as f:
                geo_config = yaml.safe_load(f)
                self.low_res_geo = Geometry(geo_config['projector'])
                self.low_res_points = self.create_low_res_space(geo_config['projector']['nVoxel'])
    def __len__(self):
        return len(self.files_names)
    def create_low_res_space(self , low_res):
        x, y, z = np.meshgrid(
            np.arange(low_res[0]),
            np.arange(low_res[1]),
            np.arange(low_res[2]),
            indexing='ij'
        )
        #pdb.set_trace()
        coords = np.stack([
            x.flatten(),
            y.flatten(),
            z.flatten()
        ], axis=1)

        # 归一化到 [0,1] 范围
        coords = coords / (np.array(low_res) - 1)
        #pdb.set_trace()
        # 添加batch维度，转换为float32类型
        coords = coords.reshape(-1, 3).astype(np.float32)

        return coords
    def load_ct(self, name):
        image, _ = sitk_load(
            os.path.join(self.data_root,  self._path_dict['image'].format(name)),
            uint8=True
        ) # float32
        return image   
    def sample_projections(self, name, n_view=2):
        # -- load projections
        with open(os.path.join(self.data_root, self._path_dict['projs'].format(name)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projs']         # uint8: [K, W, H]
            projs_max = data['projs_max'] # float
            angles = data['angles']       # float: [K,]

        if n_view is None:
            n_view = self.num_views

        # -- sample projections
        views = np.linspace(0, len(projs), n_view, endpoint=False).astype(int) # endpoint=False as the random_views is True during training, i.e., enabling view offsets.
        projs = projs[views].astype(np.float32) / 255.
        projs = projs[:, None, ...]
        angles = angles[views]
        # normalization to [-1 , 1]
        projs = (projs * 2) - 1
        # -- de-normalization
        #projs = projs * projs_max / 0.2

        return projs, angles     
    def project_points(self, points , angles):
        points_proj = []
        for a in angles:
            p = self.low_res_geo.project(points , a )
            points_proj.append(p)
        points_proj = np.stack(points_proj , axis=0)
        return points_proj
    
    def get_view_feature(self , angles , points):
        view_feature = []
        #pdb.set_trace()
        for a in angles:
            #pdb.set_trace()
            distance_ratio = self.low_res_geo.calculate_projection_distance(points , a)
            view_feature.append(distance_ratio)
            #pdb.set_trace()
        view_feature = np.stack(view_feature , axis=0)
        #pdb.set_trace()
        view_feature = np.expand_dims(view_feature , axis=-1)
        #pdb.set_trace()
        angles_feature = angles / np.pi
        angles_feature = np.expand_dims(angles_feature , axis=-1)
        angles_feature = np.repeat(angles_feature , view_feature.shape[1] , axis=-1)
        angles_feature = np.expand_dims(angles_feature , axis=-1)
        #pdb.set_trace()
        #add angels feature to view feature 
        view_feature = np.concatenate([view_feature , angles_feature] , axis=-1).astype(np.float32)
        #pdb.set_trace()
        return view_feature 

    
    def __getitem__(self, index):
        #pdb.set_trace()
        name = self.files_names[index]
        #pdb.set_trace()
        gt_idensity = self.load_ct(name)  # scale to [0,1]
        #normalization [-1,1] follow diffusion input 
        gt_idensity = gt_idensity.astype(np.float32)
        gt_idensity = (gt_idensity * 2) - 1 # to [-1,1]
        gt_idensity = np.expand_dims(gt_idensity, axis=0)
        gt_idensity = torch.from_numpy(gt_idensity)
        #pdb.set_trace()
        if self.mode == 'autoencoder':
            ret_dict = {
                    'filename': name,
                    'image': gt_idensity,
                    #'projs': projs,
                    #'angles': angles,
                }   
        else:
            projs, angles = self.sample_projections(name)
            proj_points   = self.project_points(self.low_res_points , angles)
            projs = torch.from_numpy(projs).to(torch.float32)
            proj_points = torch.from_numpy(proj_points).to(torch.float32)
            
            view_feature = self.get_view_feature(angles , self.low_res_points)
            #pdb.set_trace()
            points = deepcopy(self.low_res_points)
            points[:, :2] -= 0.5  
            points[:, 2]  = 0.5 - points[:,2]
            points *= 2 


            ret_dict = {
                'filename': name,
                'image': gt_idensity,
                'xray' : projs,
                'proj_points': proj_points,
                'coords': points,
                'view_feature' : view_feature
            }
        return ret_dict
    
def get_filesname_from_txt(txt_file_path):
    files = []
    with open(txt_file_path, 'r') as f:
        file_name = f.readlines()
        for file in file_name:
            file_name = file.strip()
            #file_path = os.path.join(base_dir, file_name)
            files.append(file_name)   

    return files



def get_pelvic_loader(config , train_mode = 'autoencoder'):
    train_files_name = get_filesname_from_txt(config['train_files_name'])
    test_files_name = get_filesname_from_txt(config['test_files_name'])
    #pdb.set_trace()
    if train_mode == 'autoencoder':
       geo_config_path = None
    else:
        geo_config_path = config['geo_config']
    train_ds = Pelivc_LatentDiffusionDataset(root_data = config['root_data'] , file_names=train_files_name , path_dict=PATH_DICT , mode=train_mode , 
                                             geo_config_path=geo_config_path)
    test_ds = Pelivc_LatentDiffusionDataset(root_data=config['root_data'], file_names=test_files_name, path_dict=PATH_DICT, mode=train_mode , 
                                            geo_config_path=geo_config_path)
    print("Dataset all training: number of data: {}".format(len(train_files_name)))
    print("Dataset all validation: number of data: {}".format(len(test_files_name)))
    
    #pdb.set_trace()
    #exp_ds = train_ds[0]
    #pdb.set_trace()
    train_loader = DataLoader(train_ds,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=config.get("num_workers", 0),
                              pin_memory=config.get("pin_memory", False))
    test_loader = DataLoader( test_ds, 
                              batch_size=1,
                              shuffle=False,
                              num_workers=config.get("num_workers", 0),
                              pin_memory=config.get("pin_memory", False))
    
    return  [train_loader , test_loader]


