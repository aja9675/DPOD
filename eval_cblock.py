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

from helper import load_obj, ADD_score, save_obj
from torchvision import transforms, utils
from create_ground_truth import get_rot_tra
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# ADD threshold Kd parameter. This is a fixed fractional scalar that's applied to the
# object's diameter to determine accuracy percentage.
#DIAMETER_THD_KD = 0.1
DIAMETER_THD_KD = 0.5

parser = argparse.ArgumentParser(description='Script to create the Ground Truth masks')
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
#correspondence_block.eval()

print("Listing all images")
list_all_images = load_obj(os.path.join(root_dir, "all_images_adr"))
testing_images_idx = load_obj(os.path.join(train_eval_dir, "test_images_indices"))

regex = re.compile(r'\d+')
upsampled = nn.Upsample(size=[240, 320], mode='bilinear', align_corners=False)
total_score = 0
print("For all %i test images..." % len(testing_images_idx))
for i in range(len(testing_images_idx)):
    if i % 100 == 0:
        print("\t %i / %i" % (i, len(testing_images_idx)))

    img_adr = list_all_images[testing_images_idx[i]]
    label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
    idx = regex.findall(os.path.split(img_adr)[1])[0]

    tra_adr = os.path.join(root_dir, label, "data/tra" + str(idx) + ".tra")
    rot_adr = os.path.join(root_dir, label, "data/rot" + str(idx) + ".rot")
    true_pose = get_rot_tra(rot_adr, tra_adr)

    test_img = cv2.imread(img_adr)
    test_img = cv2.resize(test_img, (test_img.shape[1]//2, test_img.shape[0]//2), interpolation=cv2.INTER_AREA)

    test_img = torch.from_numpy(test_img).type(torch.double)
    test_img = test_img.transpose(1, 2).transpose(0, 1)

    if len(test_img.shape) != 4:
        test_img = test_img.view(1, test_img.shape[0], test_img.shape[1], test_img.shape[2])

    # pass through correspondence block
    idmask_pred, umask_pred, vmask_pred = correspondence_block(test_img.float().cuda())

    # convert the masks to 240,320 shape
    temp = torch.argmax(idmask_pred, dim=1).squeeze().cpu()
    upred = torch.argmax(umask_pred, dim=1).squeeze().cpu()
    vpred = torch.argmax(vmask_pred, dim=1).squeeze().cpu()
    coord_2d = (temp == classes[label]).nonzero(as_tuple=True)
    if coord_2d[0].nelement() != 0:  # label is detected in the image
        coord_2d = torch.cat((coord_2d[0].view(coord_2d[0].shape[0], 1), coord_2d[1].view(coord_2d[1].shape[0], 1)), 1)
        uvalues = upred[coord_2d[:, 0], coord_2d[:, 1]]
        vvalues = vpred[coord_2d[:, 0], coord_2d[:, 1]]
        dct_keys = torch.cat((uvalues.view(-1, 1), vvalues.view(-1, 1)), 1)
        dct_keys = tuple(dct_keys.numpy())
        dct = load_obj(os.path.join(root_dir, label, "UV-XYZ_mapping"))
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
            # Need to scale the image back up to the size the correspondence map is encoded with
            mapping_2d = mapping_2d * 2
            mapping_2d[:,[0, 1]] = mapping_2d[:,[1, 0]]
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                          np.array(mapping_2d, dtype=np.float32), intrinsic_matrix, distCoeffs=None,
                                                          iterationsCount=150, reprojectionError=1.0, flags=cv2.SOLVEPNP_P3P)
            rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
            pred_pose = np.append(rot, tvecs, axis=1)

        else:  # save an empty file
            pred_pose = np.zeros((3, 4))

        diameter = np.loadtxt(os.path.join(root_dir, label, "distance.txt"))
        diameter_threshold = diameter * DIAMETER_THD_KD

        ptcld_file = os.path.join(root_dir, label, "object.xyz")
        pt_cld = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        score, avg_dist = ADD_score(pt_cld, true_pose, pred_pose, diameter_threshold)
        #print("ADD_score: %i, avg_dist: %f, thd: %f" % (score, avg_dist, diameter_threshold))
        total_score += score
        score_card[label] += score

    else:
        score_card[label] += 0

    instances[label] += 1

print("ADD Score for all testing images is: %.3f%%" % (total_score*100/len(testing_images_idx)))
print("Instances:")
print(instances)
print("Score card:")
print(score_card)
