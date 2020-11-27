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
import time

from helper import *
from torchvision import transforms, utils
from create_ground_truth import get_rot_tra
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# ADD threshold Kd parameter. This is a fixed fractional scalar that's applied to the
# object's diameter to determine accuracy percentage.
#DIAMETER_THD_KD = 0.1
DIAMETER_THD_KD = 0.5

#SHOW_RESULTS = True
SHOW_RESULTS = True
WRITE_RESULTS = False
TIMEIT = False
RANDOMIZE = True
SHOW_UV_GT = True
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

correspondence_block = UNET.UNet(n_channels=3, out_channels_id=14, out_channels_uv=256, bilinear=True)

print("Loading correspondence block")
# load the best weights from the training loop
correspondence_block_filename = os.path.join(train_eval_dir, 'correspondence_block.pt')
correspondence_block.load_state_dict(torch.load(correspondence_block_filename, map_location=torch.device('cpu')))

correspondence_block.cuda()
#correspondence_block.eval() # Causes very poor performance

print("Listing all images")
list_all_images = load_obj(os.path.join(root_dir, "all_images_adr"))
testing_images_idx = load_obj(os.path.join(train_eval_dir, "test_images_indices"))

regex = re.compile(r'\d+')
upsampled = nn.Upsample(size=[240, 320], mode='bilinear', align_corners=False)
total_score = 0
print("For all %i test images..." % len(testing_images_idx))


if WRITE_RESULTS:
    image_out_path = '/content/output_images'
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path)


if RANDOMIZE:
    np.random.shuffle(testing_images_idx)

# With fixed ADD metric, it takes a while to find a good result...
for i in testing_images_idx:

    #print("Attempting to process %i" % i)
    total_start_time = time.time()
    img_adr = list_all_images[i]

    label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
    idx = regex.findall(os.path.split(img_adr)[1])[0]

    tra_adr = os.path.join(root_dir, label + "/data/tra" + str(idx) + ".tra")
    rot_adr = os.path.join(root_dir, label + "/data/rot" + str(idx) + ".rot")
    true_pose = get_rot_tra(rot_adr, tra_adr)

    if SHOW_UV_GT:
        umask_gt_adr = os.path.join(root_dir, label, "ground_truth/Umasks/color" + str(idx) + ".png")
        vmask_gt_adr = os.path.join(root_dir, label, "ground_truth/Vmasks/color" + str(idx) + ".png")
        umask_gt = cv2.imread(umask_gt_adr, cv2.IMREAD_GRAYSCALE)
        vmask_gt = cv2.imread(vmask_gt_adr, cv2.IMREAD_GRAYSCALE)

    test_img = cv2.imread(img_adr)
    test_img_orig = deepcopy(test_img)
    test_img = cv2.resize(
        test_img, (test_img.shape[1]//2, test_img.shape[0]//2), interpolation=cv2.INTER_AREA)

    test_img = torch.from_numpy(test_img).type(torch.double)
    test_img = test_img.transpose(1, 2).transpose(0, 1)

    if len(test_img.shape) != 4:
        test_img = test_img.view(
            1, test_img.shape[0], test_img.shape[1], test_img.shape[2])

    # pass through correspondence block
    start_time = time.time()
    idmask_pred, umask_pred, vmask_pred = correspondence_block(test_img.float().cuda())
    if TIMEIT:
        print("correspondence_block: %s seconds ---" % (time.time() - start_time))

    # convert the masks to 240,320 shape
    temp = torch.argmax(idmask_pred, dim=1).squeeze().cpu()
    upred = torch.argmax(umask_pred, dim=1).squeeze().cpu()
    vpred = torch.argmax(vmask_pred, dim=1).squeeze().cpu()
    coord_2d = (temp == classes[label]).nonzero(as_tuple=True)

    print(upred.shape)

    if SHOW_RESULTS:
        upred_cpu = np.uint8(upred.detach().numpy())
        #cv2.imshow("upred_cpu", cv2.resize(upred_cpu, disp_size))
        vpred_cpu = np.uint8(vpred.detach().numpy())
        #cv2.imshow("upred_cpu", cv2.resize(upred_cpu, disp_size))
        uvpred_cpu = np.hstack((upred_cpu, vpred_cpu))
        #cv2.imshow("upred_cpu", uvpred_cpu)

        idmask_pred_cpu = np.uint8(temp.detach().numpy())
        #print(np.unique(idmask_pred_cpu))
        #idmask_pred_cpu = idmask_pred_cpu * (255 // np.max(idmask_pred_cpu))
        #print(np.unique(idmask_pred_cpu))
        #cv2.imshow("idmask", idmask_pred_cpu)

    if coord_2d[0].nelement() != 0:  # label is detected in the image

        coord_2d = torch.cat((coord_2d[0].view(coord_2d[0].shape[0], 1), coord_2d[1].view(coord_2d[1].shape[0], 1)), 1)
        uvalues = upred[coord_2d[:, 0], coord_2d[:, 1]]
        vvalues = vpred[coord_2d[:, 0], coord_2d[:, 1]]
        dct_keys = torch.cat((uvalues.view(-1, 1), vvalues.view(-1, 1)), 1)
        dct_keys = tuple(dct_keys.numpy())
        dct = load_obj(os.path.join(root_dir, label + "/UV-XYZ_mapping"))
        mapping_2d = []
        mapping_3d = []
        for count, (u, v) in enumerate(dct_keys):
            if (u, v) in dct:
                mapping_2d.append(np.array(coord_2d[count]))
                mapping_3d.append(dct[(u, v)])

        # PnP needs atleast 6 unique 2D-3D correspondences to run
        if len(mapping_2d) >= 6 or len(mapping_3d) >= 6:
            # Need to swap (row,col) to (col,row) for OpenCV points in solvePnPRansac
            mapping_2d = np.array(mapping_2d)
            mapping_2d = mapping_2d * 2
            mapping_2d[:,[0, 1]] = mapping_2d[:,[1, 0]]

            start_time = time.time()
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                          np.array(mapping_2d, dtype=np.float32), intrinsic_matrix, distCoeffs=None,
                                                          iterationsCount=300, reprojectionError=1.0, flags=cv2.SOLVEPNP_P3P)
            rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
            pred_pose = np.append(rot, tvecs, axis=1)

            orig_pred_pose = deepcopy(pred_pose)
            if TIMEIT:
                print("solvePnPRansac: %s seconds ---" % (time.time() - start_time))

        else:  # save an empty file
            print("Couldn't predict pose")
            continue


        diameter = np.loadtxt(os.path.join(root_dir, label + "/distance.txt"))
        diameter_threshold = diameter * DIAMETER_THD_KD
        ptcld_file = os.path.join(root_dir, label + "/object.xyz")
        pt_cld = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))

        start_time = time.time()
        #score, avg_dist = ADD_score(pt_cld, true_pose, orig_pred_pose, diameter * 0.1)
        score, avg_dist = ADD_score(pt_cld, true_pose, orig_pred_pose, diameter_threshold)
        print("ADD_score: %i, avg_dist: %f, thd: %f" % (score, avg_dist, diameter_threshold))
        if TIMEIT:
            print("ADD_score: %s seconds ---" % (time.time() - start_time))

        total_score += score
        score_card[label] += score

        # Skip ones that don't meet the ADD threshold
        #if not score:
        #    continue

        trans_error_pred = np.linalg.norm(true_pose[:,3] - orig_pred_pose[:,3])
        print("Translational error pred: %s: " % str(trans_error_pred))

        #display(test_img_orig)
        orig_img_filename = 'test_img' + str(i) + '.jpg'

        try:
            pose_img = deepcopy(test_img_orig)
            create_bounding_box(pose_img, true_pose, pt_cld, intrinsic_matrix, color=(0,255,0))
            create_bounding_box(pose_img, orig_pred_pose, pt_cld, intrinsic_matrix, color=(0,0,255))
            if score:
                print(true_pose)
                print(orig_pred_pose)

            axis_img = deepcopy(test_img_orig)
            draw_axis(axis_img, true_pose, intrinsic_matrix, colors=(0,255,0))
            draw_axis(axis_img, orig_pred_pose, intrinsic_matrix, colors=(255,0,0))

            pcld_img = deepcopy(test_img_orig)
            pcld_img = draw_axis(pcld_img, true_pose, intrinsic_matrix, colors=(255,0,0))
            #pcld_img = draw_point_cloud(pcld_img, true_pose, pt_cld, intrinsic_matrix, color=(0,255,0))
            #pcld_img = draw_point_cloud(pcld_img, orig_pred_pose, pt_cld, intrinsic_matrix, color=(255,0,0))

            #def ADD_vis(img, pt_cld, true_pose, pred_pose, intrinsic_matrix):
            pcld_img = ADD_vis(pcld_img, pt_cld, true_pose, orig_pred_pose, intrinsic_matrix)

        except Exception as e:
            print("Got '%s' for %i" % (e,i))
            continue


        if WRITE_RESULTS:
            pose_img_filename = 'pose_img' + str(i) + '.jpg'
            cv2.imwrite(os.path.join(image_out_path, pose_img_filename), pose_img)

        if SHOW_RESULTS:
            #if avg_dist < 500:
            if avg_dist < 100:
            #if avg_dist < 5:
                print(img_adr)
                mask_scalar = 3 # Just for visualization
                uvmask_color = color_uv(upred_cpu, vpred_cpu)
                cv2.imshow("uvmask_color", cv2.resize(uvmask_color, None,fx=mask_scalar,fy=mask_scalar))

                uvmask_color_gt = color_uv(umask_gt, vmask_gt)
                cv2.imshow("uvmask_color_gt", uvmask_color_gt)

                idmask_color = color_linemod_idmask_img(idmask_pred_cpu)
                cv2.imshow("idmask_color", cv2.resize(idmask_color, None,fx=mask_scalar,fy=mask_scalar))

                cv2.imshow("pcld_img", cv2.resize(pcld_img, disp_size))

                showImage("axis_img", cv2.resize(axis_img, disp_size))

            #showImage("pose_img", pose_img)
        
        if TIMEIT:
            print("total time: %s seconds ---" % (time.time() - total_start_time))

    else:
        print("%i Failed to find label" % i)

