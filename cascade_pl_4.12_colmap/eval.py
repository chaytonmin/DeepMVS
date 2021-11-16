from datasets import dataset_dict
from datasets.utils import save_pfm, read_pfm
import cv2
import collections
import torch
import os, shutil
from struct import *
import numpy as np
from tqdm import tqdm
import glob
from argparse import ArgumentParser

# for depth prediction
from models.mvsnet import CascadeMVSNet
from utils import load_ckpt
from inplace_abn import ABN

# for point cloud fusion
from numba import jit
from plyfile import PlyData, PlyElement

torch.backends.cudnn.benchmark = True # this increases inference speed a little

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default=',/data/',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='normal',
                        choices=['dtu', 'tanks', 'blendedmvs', 'normal'],
                        help='which dataset to train/val')
    parser.add_argument('--split', type=str, default='test',
                        help='which split to evaluate')
    parser.add_argument('--scan', type=str, default='',
                        help='specify scan to evaluate (must be in the split)')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='''use cpu to do depth inference.
                                WARNING: It is going to be EXTREMELY SLOW!
                                about 37s/view, so in total 30min/scan. 
                             ''')
    # for depth prediction
    parser.add_argument('--img_scale', type=int, default=2, help='images scales')
    parser.add_argument('--n_views', type=int, default=5,
                        help='number of views (including ref) to be used in testing')
    parser.add_argument('--depth_interval', type=float, default=160.0,
                        help='192.0 depth interval unit in mm default=2.65')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[8,32,160],
                        help='8,32,48 number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0,1.0,2.0],
                        help='1.0,1.5,3.5 1.0,2.5,5.5 1.0,2.0,4.0 depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--num_groups', type=int, default=8, choices=[1, 2, 4, 8],
                        help='number of groups in groupwise correlation, must be a divisor of 8')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[2016, 1120],
                        help='[1152, 864] resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--ckpt_path', type=str, default='./cascade_pl_4.12_colmap/ckpts/exp/_ckpt_epoch_49.ckpt',
                        help='\')
    parser.add_argument('--save_visual', default=False, action='store_true',
                        help='save depth and proba visualization or not')
    parser.add_argument('--depth_dir', type=str, default='./results/depth',
                        help='depth path to save')       
    parser.add_argument('--point_dir', type=str, default='./results/point',
                        help='point path to save')
    parser.add_argument('--list_dir', type=str, default='data/lists/testing_list.txt',
                        help='lists dir')
    parser.add_argument('--conf', type=float, default=0.999,
                        help='min confidence for pixel to be valid')                                                

    return parser.parse_args()


def decode_batch(batch):
    imgs = batch['imgs']
    proj_mats = batch['proj_mats']
    proj_ls = batch['proj_ls']
    init_depth_min = batch['init_depth_min'].item()
    init_depth_max = batch['init_depth_max'].item()
    depth_interval = batch['depth_interval'].item()
    view_ids = batch['view_ids']
    scan, vid = batch['scan_vid']
    return imgs, proj_mats, init_depth_min, depth_interval, \
           scan, vid, init_depth_max

def read_proj_inex_mat(dataset_name, dataset, scan, vid):
    if dataset_name == 'dtu':
        return dataset.proj_mats[vid][2][2].numpy()
    if dataset_name in ['tanks', 'blendedmvs', 'normal']:
        return dataset.proj_mats[scan][vid][2][2].numpy()


# define read_image and read_proj_mat for each dataset

def read_image(dataset_name, root_dir, scan, vid):
    if dataset_name == 'dtu':
        return cv2.imread(os.path.join(root_dir,
                    f'Rectified/{scan}/rect_{vid+1:03d}_3_r5000.png'))
    if dataset_name == 'tanks':
        return cv2.imread(os.path.join(root_dir, scan,
                    f'images/{vid:08d}.jpg'))
    if dataset_name == 'blendedmvs':
        return cv2.imread(os.path.join(root_dir, scan,
                    f'blended_images/{vid:08d}.jpg'))
    if dataset_name == 'normal':
        return cv2.imread(os.path.join(root_dir, scan,
                    f'images_mvsnet/{vid:08d}.jpg'))


def read_proj_mat(dataset_name, dataset, scan, vid):
    if dataset_name == 'dtu':
        return dataset.proj_mats[vid][0][0].numpy()
    if dataset_name in ['tanks', 'blendedmvs', 'normal']:
        return dataset.proj_mats[scan][vid][0][0].numpy()

def fake_gipuma_normal(depth_image):
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image)

    return normal_image

if __name__ == "__main__":
    args = get_opts()
    img_scale = args.img_scale
    # set the resized size
    
    list_dir = args.list_dir
    with open(list_dir, "r") as f:
      scan = f.readline()
    img_dir = os.path.join(args.root_dir, f'{scan}/colmap/images')
    first_img_path = os.listdir(img_dir)[0]
    first_img = cv2.imread(os.path.join(img_dir,first_img_path))
    h,w = first_img.shape[:2]
    print('w,h:',w,h)

    new_w = int(w/img_scale)
    new_h = int(h/img_scale)
    args.img_wh = [new_w, new_h]
    print('args.img_wh:',args.img_wh)
 
    print('args.depth_interval:',args.depth_interval)

    dataset = dataset_dict[args.dataset_name] \
                (args.list_dir, args.root_dir, args.split,
                 n_views=args.n_views, depth_interval=args.depth_interval,
                 img_wh=tuple(args.img_wh))

    if args.scan:
        scans = [args.scan]
    else: # evaluate on all scans in dataset
        scans = dataset.scans
    
    # Create depth estimation and probability for each scan
    model = CascadeMVSNet(n_depths=args.n_depths,
                          interval_ratios=args.interval_ratios,
                          num_groups=args.num_groups,
                          norm_act=ABN)
    device = 'cpu' if args.cpu else 'cuda:0'
    model.to(device)
    load_ckpt(model, args.ckpt_path)
    model.eval()

    depth_dir = args.depth_dir
    print('Creating depth and confidence predictions...')
    if args.scan:
        data_range = [i for i, x in enumerate(dataset.metas) if x[0] == args.scan]
    else:
        data_range = range(len(dataset))

    for i in tqdm(data_range):
        imgs, proj_mats, init_depth_min, depth_interval, \
            scan, vid, init_depth_max = decode_batch(dataset[i])
        
        os.makedirs(depth_dir, exist_ok=True)

        with torch.no_grad():
            imgs = imgs.unsqueeze(0).to(device)
            proj_mats = proj_mats.unsqueeze(0).to(device)
            results = model(imgs, proj_mats, init_depth_min, depth_interval)
        
        depth = results['depth_0'][0].cpu().numpy()
        depth = np.nan_to_num(depth) # change nan to 0
        depth[depth>init_depth_max] = 0
        depth[depth<init_depth_min] = 0
        #depth /= 20
        proba = results['confidence_2'][0].cpu().numpy() # NOTE: this is 1/4 scale!
        proba = np.nan_to_num(proba) # change nan to 0
        save_pfm(os.path.join(depth_dir, f'depth_{vid:04d}.pfm'), depth)
        save_pfm(os.path.join(depth_dir, f'proba_{vid:04d}.pfm'), proba)
        
        if args.save_visual:
            mi = np.min(depth[depth>0])
            ma = np.max(depth)
            depth = (depth-mi)/(ma-mi+1e-8)
            depth = (255*depth).astype(np.uint8)
            depth_img = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(depth_dir, f'depth_visual_{vid:04d}.jpg'),
                        depth_img)
            cv2.imwrite(os.path.join(depth_dir, f'proba_visual_{vid:04d}.jpg'),
                        (255*(proba>args.conf)).astype(np.uint8))
        del imgs, proj_mats, results
    del model
    torch.cuda.empty_cache()
