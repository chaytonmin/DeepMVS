#!/usr/bin/env python


from __future__ import print_function

import os
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import scipy.io
import urllib

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    
    ##print("scale_camera--new_cam")
    ##print(new_cam, new_cam.shape, scale)
    # focal: 
    new_cam[0][0] = cam[0][0] * scale
    new_cam[1][1] = cam[1][1] * scale
    # principle point:
    new_cam[0][2] = cam[0][2] * scale
    new_cam[1][2] = cam[1][2] * scale
    return new_cam

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        ##print("preprcoss-scale_image-image,scale")
        ##print(image.shape,scale)
        #return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        #return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def scale_mvs_input(images, cams, depth_image=None, scale=1,view_num=5):
    """ resize input to fit into the memory """
    new_images = []
    new_cams=[]
    for view in range(view_num):
        ##print("scale_mvs_input-images[view], scale=scale")
        ##print(images[view].shape, scale)
        #images[view] = scale_image(images[view], scale=scale)
        new_images.append(scale_image(images[view], scale=scale))
        ##print("cams[view], scale=scale")
        ##print(cams[view],cams.shape)
        ##cams[view] = scale_camera(cams[view], scale=scale)
        new_cams.append(scale_camera(cams[view], scale=scale))
        ##print("cams[view]")
        ##print(cams[view], cams[view].shape)
    new_images = np.array(new_images)
    if depth_image is None:
        #return images, cams
        return new_images, new_cams
    else:
        #depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        #return images, cams, depth_image
        depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        return new_images, cams, depth_image

def crop_mvs_input(images, cams, depth_image=None,view_num=5,max_h=1200,max_w=1600,base_image_size=8):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    
    new_images = []
    # crop images and cameras
    for view in range(view_num):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w
        if new_h > max_h:
            new_h = max_h
        else:
            new_h = int(math.ceil(h /base_image_size) * base_image_size)
        if new_w > max_w:
            new_w = max_w
        else:
            new_w = int(math.ceil(w /base_image_size) * base_image_size)
        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        # print(" start_h,start_w,finish_h,finish_w")
        # print(start_h,start_w,finish_h,finish_w)
     
        ##images[view] = images[view][start_h:finish_h, start_w:finish_w]
        new_images.append(images[view][start_h:finish_h, start_w:finish_w])
        cams[view][0][2] = cams[view][0][2] - start_w
        cams[view][1][2] = cams[view][1][2] - start_h

    new_images = np.stack(new_images)
    # crop depth image
    if not depth_image is None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        ##return images, cams, depth_image
        return new_images, cams, depth_image
    else:
        ##return images, cams
        return new_images, cams

