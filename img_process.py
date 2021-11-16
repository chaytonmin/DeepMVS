# -*- coding: utf-8 -*-
import os, cv2, sys
import argparse
import shutil
import numpy as np
import glob
import multiprocessing as mp

def resize_img(inputs):
    i, path, image_resize_dir, img_scale = inputs
    #print('path:',path)
    img = cv2.imread(path)
    h,w = img.shape[:2]
    max_h = 1300
    max_w = 1300
    base = 32
    if h > max_h or w > max_w:
        scale = 1.0 * max_h / h
        if scale * w > max_w:
            scale = 1.0 * max_w / w
        new_w, new_h = int(scale * w // base * base), int(scale * h // base * base)
    else:
        new_w, new_h = int(1.0 * w // base * base), int(1.0 * h // base * base)
            
    #print('new_w,new_h:',new_w,new_h)
    img = cv2.resize(img,(img_scale*new_w,img_scale*new_h))

    imidx = '%08d.png' % (i+1)
    cv2.imwrite(image_resize_dir+imidx,img)


def main():

    parser = argparse.ArgumentParser(description='get main part')
    parser.add_argument('--img_scale', type=int, default=2, help='images scales')
    parser.add_argument('--image_dir', default='./', type=str, help='img dir.')
    parser.add_argument('--image_resize_dir', default='./', type=str, help='img resize dir.')
    
    args = parser.parse_args()
   
    image_dir = args.image_dir 
    image_resize_dir = args.image_resize_dir
    img_scale = args.img_scale

    os.makedirs(image_resize_dir, exist_ok=True)

    img_name_list = glob.glob(image_dir + '*')

    queue = []
    for i in range(len(img_name_list)):
        queue.append((i,img_name_list[i],image_resize_dir,img_scale))   
    print('start image resize:')
    p = mp.Pool(processes=mp.cpu_count())
    p.map(resize_img, queue)
    print('image resize Done!')


if __name__ == "__main__":
    main()
