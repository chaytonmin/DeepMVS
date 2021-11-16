from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, inverse_depth=False,
                 origin_size=False, light_idx=-1, image_scale=0.25, reverse=False, both=True, fix_range=False, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth
        self.origin_size = origin_size
        self.light_idx=light_idx
        self.image_scale = image_scale # use to resize image
        self.reverse = reverse
        self.both = both
        self.fix_range = fix_range
        print('dataset: inverse_depth {}, origin_size {}, light_idx:{}, image_scale:{}, reverse: {}, both: {}'.format(
                    self.inverse_depth, self.origin_size, self.light_idx, self.image_scale, self.reverse, self.both))
        
        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if self.light_idx == -1:
                        for light_idx in range(7):
                            if self.both:
                                metas.append((scan, light_idx, ref_view, src_views, 1)) # add 1, 0 for reverse depth
                            metas.append((scan, light_idx, ref_view, src_views, 0))
                    else:
                        if self.both:
                            metas.append((scan, self.light_idx, ref_view, src_views, 1))
                        metas.append((scan, self.light_idx, ref_view, src_views, 0))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        #return len(self.metas)
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        if self.image_scale == 0.5: # origin: 0.25
            intrinsics[:2, :] *= 2
        elif self.image_scale == 1.0:
            intrinsics[:2, :] *= 4
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        if self.image_scale != 1.0:
            w, h = img.size
            img = img.resize((int(self.image_scale * w), int(self.image_scale*h))) # origin: 0.25
        # scale 0~255 to 0~1
        #np_img = np.array(img, dtype=np.float32) / 255. # origin version on 2020/02/20
        #return np_img
        return self.center_img(np.array(img, dtype=np.float32))	

    def center_img(self, img): # this is very important for batch normalization
        img = img.astype(np.float32)
        var = np.var(img, axis=(0,1), keepdims=True)
        mean = np.mean(img, axis=(0,1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)
        

    def __getitem__(self, idx):
        
        #print('idx: {}, flip_falg {}'.format(idx, flip_flag))
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views, flip_flag = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            if self.image_scale == 1.0:
                depth_filename = os.path.join(self.datapath, '../640_depth/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
                mask_filename = os.path.join(self.datapath, '../640_depth/{}/depth_map_{:0>4}_mask.png'.format(scan, vid))
            elif self.image_scale == 0.5:
                depth_filename = os.path.join(self.datapath, '../320_depth/{}/depth_map_{:0>4}_4.pfm'.format(scan, vid))
                mask_filename = os.path.join(self.datapath, '../320_depth/{}/depth_map_{:0>4}_mask_4.png'.format(scan, vid))
            else:
                depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
                mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))

            
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            if i == 0:
                depth_name = depth_filename
            #print('debug in dtu_yao', i, depth_filename)
            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                if not self.fix_range:
                    depth_end = depth_interval * (self.ndepths-1) + depth_min # sample: 0:n-1
                else:
                    depth_end = 935 # pre-defined in DTU dataset
                if self.inverse_depth: #slice inverse depth
                    print('inverse depth')
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_end, self.ndepths)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)
                else:
                    depth_values = np.linspace(depth_min, depth_end, self.ndepths)
                    depth_values = depth_values.astype(np.float32)
                    
                #mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)
                #mask = np.array((depth > depth_min+depth_interval) & (depth < depth_min+(self.ndepths-2)*depth_interval), dtype=np.float32)
                mask = np.array((depth >= depth_min) & (depth <= depth_end), dtype=np.float32)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        if (flip_flag and self.both) or (self.reverse and not self.both):
            depth_values = np.array([depth_values[len(depth_values)-i-1]for i in range(len(depth_values))])
        
        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "depth_values": depth_values, # generate depth index
                "mask": mask,
                "depth_interval": depth_interval,
                'name':depth_name,}
