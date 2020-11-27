#!/usr/bin/env python3

import sys
import os
import re
import cv2
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import unet_model as UNET
from copy import deepcopy

from helper import *
from torchvision import transforms, utils
from create_ground_truth import get_rot_tra
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

RANDOMIZE = True

disp_size = (1600, 1200)

parser = argparse.ArgumentParser(
    description='Script to create the Ground Truth masks')
parser.add_argument("root_dir", help="path to dataset directory (LineMOD_Dataset)")
parser.add_argument("train_eval_dir", help="path to dir to store training run specific info")
args = parser.parse_args()

root_dir = args.root_dir
train_eval_dir = args.train_eval_dir

classes = {'ape': 1, 'benchviseblue': 2, 'cam': 3, 'can': 4, 'cat': 5, 'driller': 6,
           'duck': 7, 'eggbox': 8, 'glue': 9, 'holepuncher': 10, 'iron': 11, 'lamp': 12, 'phone': 13}

score_card = {'ape': 0, 'benchviseblue': 0, 'cam': 0, 'can': 0, 'cat': 0, 'driller': 0,
              'duck': 0, 'eggbox': 0, 'glue': 0, 'holepuncher': 0, 'iron': 0, 'lamp': 0, 'phone': 0}

instances = {'ape': 0, 'benchviseblue': 0, 'cam': 0, 'can': 0, 'cat': 0, 'driller': 0,
             'duck': 0, 'eggbox': 0, 'glue': 0, 'holepuncher': 0, 'iron': 0, 'lamp': 0, 'phone': 0}

transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                transforms.Resize(size=(224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899  # Intrinsic Parameters of the Camera
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])


print("Listing all images")
list_all_images = load_obj(os.path.join(root_dir, "all_images_adr"))
testing_images_idx = load_obj(os.path.join(train_eval_dir, "test_images_indices"))

regex = re.compile(r'\d+')

print("For all %i test images..." % len(testing_images_idx))

if RANDOMIZE:
    np.random.shuffle(testing_images_idx)

# With fixed ADD metric, it takes a while to find a good result...
for i in testing_images_idx:

    #print("Attempting to process %i" % i)
    img_adr = list_all_images[i]
    print(img_adr)

    label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
    idx = regex.findall(os.path.split(img_adr)[1])[0]

    test_img = cv2.imread(img_adr)
    #print(test_img.shape)
    cv2.imshow("test_img", cv2.resize(test_img, disp_size))

    tra_adr = os.path.join(root_dir, label + "/data/tra" + str(idx) + ".tra")
    rot_adr = os.path.join(root_dir, label + "/data/rot" + str(idx) + ".rot")
    true_pose = get_rot_tra(rot_adr, tra_adr)

    # Read GTs
    idmask_gt_adr = os.path.join(root_dir, label, "ground_truth/IDmasks/color" + str(idx) + ".png")
    umask_gt_adr = os.path.join(root_dir, label, "ground_truth/Umasks/color" + str(idx) + ".png")
    vmask_gt_adr = os.path.join(root_dir, label, "ground_truth/Vmasks/color" + str(idx) + ".png")
    idmask_gt = cv2.imread(idmask_gt_adr, cv2.IMREAD_GRAYSCALE)
    umask_gt = cv2.imread(umask_gt_adr, cv2.IMREAD_GRAYSCALE)
    vmask_gt = cv2.imread(vmask_gt_adr, cv2.IMREAD_GRAYSCALE)
    umask_gt = cv2.resize(umask_gt, (test_img.shape[1], test_img.shape[0]), interpolation=cv2.INTER_AREA)
    vmask_gt = cv2.resize(vmask_gt, (test_img.shape[1], test_img.shape[0]), interpolation=cv2.INTER_AREA)
    print(umask_gt.shape)
    uvmask_color_gt = color_uv(umask_gt, vmask_gt)
    #print(uvmask_color_gt.shape)

    idmask_color = color_linemod_idmask_img(idmask_gt)
    cv2.imshow("idmask_color", idmask_color)
    cv2.imshow("uvmask_color_gt", uvmask_color_gt)

    uv_xyz_dct = load_obj(os.path.join(root_dir, label + "/UV-XYZ_mapping"))

    # Get 2D coords from ID mask (row, col)
    coord_2d = np.argwhere(idmask_gt == classes[label])

    # Check that the points are correct
    if 0:
        for pt in coord_2d:
            # Note OpenCV points are (col,row)
            test_img = cv2.circle(test_img, (pt[1],pt[0]), 1, (0,255,0), 1)
        showImage("test_imgcirc", cv2.resize(test_img, disp_size))

    # At each coord, find the U and V values (row, col)
    uvalues = umask_gt[coord_2d[:,0], coord_2d[:,1]]
    vvalues = vmask_gt[coord_2d[:,0], coord_2d[:,1]]
    uv_values = np.vstack((uvalues,vvalues)).T

    mapping_2d = []
    mapping_3d = []
    for count, (u, v) in enumerate(uv_values):
        if (u, v) in uv_xyz_dct:
            mapping_2d.append(np.array(coord_2d[count]))
            mapping_3d.append(uv_xyz_dct[(u, v)])

    # Need to swap (row,col) to (col,row) for OpenCV points in solvePnPRansac
    mapping_2d = np.array(mapping_2d)
    mapping_2d[:,[0, 1]] = mapping_2d[:,[1, 0]]

    # PnP needs atleast 6 unique 2D-3D correspondences to run
    if len(mapping_2d) >= 6 or len(mapping_3d) >= 6:
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                      np.array(mapping_2d, dtype=np.float32), intrinsic_matrix, distCoeffs=None,
                                                      iterationsCount=150, reprojectionError=1.0, flags=cv2.SOLVEPNP_P3P)
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        pred_pose = np.append(rot, tvecs, axis=1)

    else:  # save an empty file
        print("Couldn't predict pose")
        continue

    trans_error_pred = np.linalg.norm(true_pose[:,3] - pred_pose[:,3])
    print("Translational error pred: %s: " % str(trans_error_pred))

    try:
        #print(true_pose)
        #print(pred_pose)
        axis_img = deepcopy(test_img)
        draw_axis(axis_img, true_pose, intrinsic_matrix, colors=(0,255,0))
        draw_axis(axis_img, pred_pose, intrinsic_matrix, colors=(255,0,0))

    except Exception as e:
        print("Got '%s' for %i" % (e,i))
        continue

    showImage("axis_img", cv2.resize(axis_img, disp_size))
