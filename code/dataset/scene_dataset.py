import jittor as jt
import os
import numpy as np
from utils import rend_util
from jittor.dataset import Dataset
import utils.general as utils

class SceneDataset(Dataset):
    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 scan_id=65,
                 cam_file=None):
        
        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        
        self.train_cameras = train_cameras
        
        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(utils.glob(mask_dir))
        
        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(jt.array(intrinsics).float())
            self.pose_all.append(jt.array(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(jt.array(rgb).float())

        self.object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(jt.array(object_mask).bool())
            
    def __getitem__(self, index):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = jt.array(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        
        sample = {
            "object_mask": self.object_masks[index],
            "uv": uv,
            "intrinsics": self.intrinsics_all[index],
        }
        
        ground_truth = {
            "rgb": self.rgb_images[index]
        }
        
        if not self.train_cameras:
            sample["pose"] = self.pose_all[index]
            
        return idx, sample, ground_truth
        
    def __len__(self):
        return self.n_images
