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
import torch.nn.functional as F
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

#SHOW_RESULTS = True
SHOW_RESULTS = True
WRITE_RESULTS = False
TIMEIT = False
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

instances = {'ape': 0, 'benchviseblue': 0, 'cam': 0, 'can': 0, 'cat': 0, 'driller': 0,
             'duck': 0, 'eggbox': 0, 'glue': 0, 'holepuncher': 0, 'iron': 0, 'lamp': 0, 'phone': 0}

transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                transforms.Resize(size=(224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

correspondence_block = UNET.UNet(n_channels=3, out_channels_id=14, bilinear=True)

# load the best weights from the training loop
correspondence_block_filename = os.path.join(train_eval_dir, 'correspondence_block.pt')
print("Loading correspondence block: %s" % correspondence_block_filename)
correspondence_block.load_state_dict(torch.load(correspondence_block_filename, map_location=torch.device('cpu')))

correspondence_block.cuda()
#correspondence_block.train()
# Results are very poor with eval()
#correspondence_block.eval()

print("Listing all images")
list_all_images = load_obj(os.path.join(root_dir, "all_images_adr"))

if 0: # Hack for initial debugging
    testing_images_idx = load_obj(os.path.join(train_eval_dir, "train_images_indices"))

    for i in testing_images_idx:
        img_adr = list_all_images[i]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        instances[label] += 1
        #unique, counts = np.unique(testing_images_idx, return_counts=True)
    print(instances)
else:
    testing_images_idx = load_obj(os.path.join(train_eval_dir, "test_images_indices"))

regex = re.compile(r'\d+')
upsampled = nn.Upsample(size=[240, 320], mode='bilinear', align_corners=False)
total_score = 0
print("For all %i test images..." % len(testing_images_idx))


if RANDOMIZE:
    np.random.shuffle(testing_images_idx)

# With fixed ADD metric, it takes a while to find a good result...
for i in testing_images_idx:

    #print("Attempting to process %i" % i)
    total_start_time = time.time()
    img_adr = list_all_images[i]

    label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
    idx = regex.findall(os.path.split(img_adr)[1])[0]

    test_img = cv2.imread(img_adr)
    test_img_orig = deepcopy(test_img)
    test_img = cv2.resize(test_img, (test_img.shape[1]//2, test_img.shape[0]//2), interpolation=cv2.INTER_AREA)

    showImage("test_img", cv2.resize(test_img, disp_size), hold=False)

    test_img = torch.from_numpy(test_img).type(torch.float)
    test_img = test_img.transpose(1, 2).transpose(0, 1)

    if len(test_img.shape) != 4:
        test_img = test_img.view(1, test_img.shape[0], test_img.shape[1], test_img.shape[2])

    # pass through correspondence block
    start_time = time.time()

    idmask_pred = correspondence_block(test_img.cuda())
    if TIMEIT:
        print("correspondence_block: %s seconds ---" % (time.time() - start_time))

    show_predictions_tiled(idmask_pred, hold=False)

    # convert the masks to 240,320 shape
    idmask_pred = idmask_pred.squeeze()
    idmask_max_pred = torch.argmax(idmask_pred, dim=0).cpu()
    coord_2d = (idmask_max_pred == classes[label]).nonzero(as_tuple=True)

    idmask_pred_cpu = np.uint8(idmask_max_pred.detach().numpy())
    mask_scalar = 3 # Just for visualization
    idmask_color = color_linemod_idmask_img(idmask_pred_cpu)
    showImage("idmask_color", cv2.resize(idmask_color, None,fx=mask_scalar,fy=mask_scalar))

    if coord_2d[0].nelement() != 0:  # label is detected in the image
        #print("Success: %s" % img_adr)
        #showImage("test_img_orig", cv2.resize(test_img_orig, disp_size))
        pass

    else:
        print("%i Failed to find label" % i)
