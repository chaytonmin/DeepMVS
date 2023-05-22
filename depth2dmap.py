import os, shutil
from struct import *
import numpy as np
from argparse import ArgumentParser
import cv2,glob
import re

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/mnt/md0/codes/mvs/data/',
                        help='root directory of dtu dataset')
    parser.add_argument('--img_scale', type=int, default=2, help='images scales')
    parser.add_argument('--depth_dir', type=str, default='/mnt/md0/codes/mvs/results/depth',
                        help='depth path to save')
    parser.add_argument('--list_dir', type=str, default='data/lists/testing_list.txt',
                        help='lists dir')      

    return parser.parse_args()

def fake_gipuma_normal(depth_image):
    # generate the fake normal for depth fusion
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

def read_pfm(filename):
    # read the pfm depth
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

if __name__ == "__main__":
    args = get_opts()
    img_scale = args.img_scale
    depth_dir = args.depth_dir

    nIDs = 5  #num of neighors

    # get img size
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

    # copy img 
    img_dir = os.path.join(depth_dir,'../images/')
    os.makedirs(img_dir, exist_ok=True)
    img_resize_list = glob.glob(os.path.join(args.root_dir,f'{scan}/images_mvsnet/') + '*')
    for path in img_resize_list:
        shutil.copy(path,img_dir)
    
    # get the img neighbors
    src_views_dict = {}
    with open(os.path.join(args.root_dir, scan, "cams/pair.txt")) as f:
        num_viewpoint = int(f.readline())
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            src_views_dict[ref_view] = src_views[:nIDs]
    
    #read colmap camera id
    id = []
    #read colmap image id
    id = []
    with open(os.path.join(args.root_dir, f'{scan}/colmap/sparse/images.txt')) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if len(line) < 0 or line[0] == "#":
                continue
            elems = line.split()
            image_id = int(elems[0])
            id.append(image_id)
            line = f.readline()

    #follow the colmap camera id to write depth
    print('id:',id)
    k = 0

    for vid in id:
        if 1: 
            print('k,vid:',k,vid)
            depth,_ = read_pfm(os.path.join(depth_dir, f'depth_est_0/{vid:08d}.pfm'))
            proba,_ = read_pfm(os.path.join(depth_dir, f'confidence_0/{vid:08d}.pfm'))
            print(depth.shape)

            # write to dmap
            type = 2 # 0: only has depthmap, 2:has depthmap, normalmap, confMap
            imageWidth = int(new_w/2*img_scale)  # image width and height as the above deep model
            imageHeight = int(new_h/2*img_scale)
            depthWidth = int(new_w/2*img_scale)
            depthHeight = int(new_h/2*img_scale)
            with open(os.path.join(args.root_dir, f'{scan}/cams/{vid:08d}_cam.txt')) as f:
                lines = [line.rstrip() for line in f.readlines()]
                dMin = float(lines[11].split()[0])# depth range
                dMax = float(lines[11].split()[3])
                # cameras extrinsics: line [1,5), 4x4 matrix
                extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
                extrinsics = extrinsics.reshape((4, 4))
                # cameras intrinsics: line [7-10), 3x3 matrix
                intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
                intrinsics = intrinsics.reshape((3, 3))

            nFileNameSize = 19
            FileName = 'images/'+f'{vid:08d}.png' 
            
            IDs = src_views_dict[vid][:nIDs] # images neighors
            print('IDs:',IDs)
        
            K = intrinsics  #cameras intrincs
            K[0,:] *=img_scale
            K[1,:] *=img_scale
        
            R = extrinsics[:,:3] #cameras extrincs 
            C = extrinsics[:,3].T #cameras extrincs 
       
            K = K.reshape((1,-1),order='A')
            R = R.reshape((1,-1),order='A')
            C = C.reshape((1,-1),order='A')
            # dmap dir
            
            path = os.path.join(depth_dir, f'../depth{k:04d}.dmap') # depthmap id from 0, not as img id
            k +=1
            print('path:',path)
            # depth map
            depthMap = depth#cv2.resize(depth, None, fx=2*img_scale, fy=2*img_scale, interpolation=cv2.INTER_LINEAR)
            # normal map
            normalMap = fake_gipuma_normal(depthMap)
            # confMap
            confMap = cv2.resize(proba, None, fx=2*img_scale, fy=2*img_scale, interpolation=cv2.INTER_LINEAR)
            #print('depthMap:',depthMap)
            #print('normalMap:',normalMap)
            #print('confMap:',confMap)

            with open(path, "wb") as fid:
                # write header as openMVS interface.h
                fid.write(pack('3c', b'D',b'R',b'\x07'))
                fid.write(pack('<B', type))
                fid.write(pack('4I', imageWidth,imageHeight,depthWidth,depthHeight))
                fid.write(pack('2f', dMin,dMax)) 
                fid.write(pack('H', nFileNameSize)) # lengh of img
                for i in range(nFileNameSize):
                    fid.write(pack('s', bytes(FileName[i].encode('utf-8')))) #img_dir

                # write neighbors ids
                fid.write(pack('I', nIDs)) #num of neighbor
                for i in range(nIDs):
                    if i<=len(IDs):
                        fid.write(pack('I', IDs[i])) #IDs of neighbors
                    else:
                        fid.write(pack('I', IDs[0])) 

                #fid.write(pack(f'{nIDs}I', IDs[0],IDs[1],IDs[2],IDs[3])) #IDs of neighbors
                fid.write(pack('d', nIDs))

                # write cameras K R C
                for i in range(9):
                    fid.write(pack('d', K[0,i]))
                for i in range(9):
                    fid.write(pack('d', R[0,i]))
                for i in range(3):    
                    fid.write(pack('d', C[0,i]))

                # write depthMap
                depthMap.tofile(fid,format="%f")
                normalMap.tofile(fid,format="%f")
                confMap.tofile(fid,format="%f")