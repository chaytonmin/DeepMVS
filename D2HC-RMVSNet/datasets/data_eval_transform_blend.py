from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *

from datasets.preprocess import *

# Test any dataset with scale and center crop

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, inverse_depth=True,
                adaptive_scaling=True, max_h=1200,max_w=1600,sample_scale=1,base_image_size=8,**kwargs):
        super(MVSDataset, self).__init__()
        
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth

        self.adaptive_scaling=adaptive_scaling
        self.max_h=max_h
        self.max_w=max_w
        self.sample_scale=sample_scale
        self.base_image_size=base_image_size
        
        assert self.mode == "test"
        self.metas = self.build_list()
        print('Data Loader : data_eval_transform **************' )

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # TODO Scale
        #intrinsics[:2, :] /= 4
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    # def read_img(self, filename):
    #     img = Image.open(filename)
    #     np_img = np.array(img, dtype=np.float32) / 255.
    #     return np_img

    def read_img(self, filename):
        img = Image.open(filename)
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
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        cams=[]
        extrinsics_list=[]

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            cams.append(intrinsics)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics_list.append(extrinsics)
            
            if i == 0:  # reference view
                if self.inverse_depth: #slice inverse depth
                    print('Process {} inverse depth'.format(idx))
                    depth_end = depth_interval * (self.ndepths-1) + depth_min # wether depth_end is this
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_end, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)
                else:
                    depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                            dtype=np.float32) # the set is [)
                    depth_end = depth_interval * self.ndepths + depth_min
                # depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                #                          dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2]) # B,C,H,W
        #proj_matrices = np.stack(proj_matrices)
        
        ##TO DO determine a proper scale to resize input
        resize_scale = 1
        if self.adaptive_scaling:
            h_scale = 0
            w_scale = 0       
            for view in range(self.nviews):
                height_scale = float(self.max_h) / imgs[view].shape[1]
                width_scale = float(self.max_w) / imgs[view].shape[2]
                if height_scale > h_scale:
                    h_scale = height_scale
                if width_scale > w_scale:
                    w_scale = width_scale
            if h_scale > 1 or w_scale > 1:
                print ("max_h, max_w should < W and H!")
                exit(-1)
            resize_scale = h_scale
            if w_scale > h_scale:
                resize_scale = w_scale
        
        imgs = imgs.transpose(0,2,3,1)
        
        scaled_input_imgs, scaled_input_cams = scale_mvs_input(imgs, cams, scale=resize_scale, view_num=self.nviews)
              
        #TO DO crop to fit network
        croped_imgs, croped_cams = crop_mvs_input(scaled_input_imgs, scaled_input_cams,view_num=self.nviews,
                    max_h=self.max_h,max_w=self.max_w,base_image_size=self.base_image_size)
                    
        croped_imgs = croped_imgs.transpose(0,3,1,2)


        new_proj_matrices = []
        for id in range(self.nviews):
            proj_mat = extrinsics_list[id]#.copy()
            # Down Scale
            #croped_cams[id][:2,:] /= 4
            proj_mat[:3, :4] = np.matmul(croped_cams[id], proj_mat[:3, :4])
            new_proj_matrices.append(proj_mat)

        new_proj_matrices = np.stack(new_proj_matrices)

        return {"imgs": croped_imgs,
                "proj_matrices": new_proj_matrices,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    #datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, adaptive_scaling=True, max_h=1200,max_w=1600,sample_scale=0.25
    dataset = MVSDataset("/data/yhw/pytorch_dtu/dtu_test/", '../lists/dtu/test.txt', 'test', 5,
                         192,1.06,adaptive_scaling=True,max_h=800,max_w=1200,sample_scale=1,base_image_size=8)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))
