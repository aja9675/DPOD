#!/usr/bin/env python3

import sys
import os
import re
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from helper import save_obj, load_obj
from pose_block import initial_pose_estimation
from create_renderings import create_refinement_inputs
from pose_refinement import train_pose_refinement
from correspondence_block import train_correspondence_block
from create_ground_truth import create_GT_masks, create_UV_XYZ_dictionary, \
		ground_truth_dir_structure, test_dir_structure

# Update classes if you're using a different dataset
classes = {'ape': 1, 'benchviseblue': 2, 'cam': 3, 'can': 4, 'cat': 5, 'driller': 6,
			'duck': 7, 'eggbox': 8, 'glue': 9, 'holepuncher': 10, 'iron': 11, 'lamp': 12, 'phone': 13}


parser = argparse.ArgumentParser(description='Script to create the Ground Truth masks')
parser.add_argument("root_dir", help="path to dataset directory (LineMOD_Dataset)")
parser.add_argument("bgd_dir", help="path to background images dataset directory (val2017)")
parser.add_argument("train_eval_dir", help="path to dir to store training run specific info")
parser.add_argument("--split", default=0.15, help="train:test split ratio")
parser.add_argument("--randomseed", default=69, help="train:test split random seed")
args = parser.parse_args()

root_dir = args.root_dir
background_dir = args.bgd_dir
train_eval_dir = args.train_eval_dir

# Pickling this eliminates the need to walk the dir which takes a while, especially
# on Google Colab when using a mounted drive
if os.path.exists(os.path.join(root_dir, "all_images_adr.pkl")):
	print("all_images_adr.pkl found. Assuming GT exists")
	gt_exists = True
	list_all_images = load_obj(os.path.join(root_dir, "all_images_adr"))
else:
	gt_exists = False
	list_all_images = []
	for root, dirs, files in os.walk(root_dir):
	    for file in files:
	        if file.endswith(".jpg"):  # images that exist
	            list_all_images.append(os.path.join(root, file))

if os.path.exists(os.path.join(train_eval_dir, "train_images_indices.pkl")):
	sys.exit("train_images_indices.pkl found. Nothing to do.")

num_images = len(list_all_images)
indices = list(range(num_images))
np.random.seed(args.randomseed)
np.random.shuffle(indices)
split = int(np.floor(float(args.split) * num_images))
train_idx, test_idx = indices[:split], indices[split:]

print("Total number of images: ", num_images)
print(" Total number of training images: ", len(train_idx))
print(" Total number of testing images: ", len(test_idx))

# Save the test/train split to the unique training dir
if not os.path.exists(train_eval_dir):
	os.makedirs(train_eval_dir)
save_obj(train_idx, os.path.join(train_eval_dir, "train_images_indices"))
save_obj(test_idx, os.path.join(train_eval_dir, "test_images_indices"))
# Create test/eval dir structure
test_dir_structure(train_eval_dir, classes)

# Only need to do this once on your dataset, it will generate ALL GT data,
# not only what you need for your train/val split. This takes longer initially
# but speeds up later development.
if not gt_exists:
	# Save all images list to the root dir
	save_obj(list_all_images, os.path.join(root_dir, "all_images_adr"))

	# Create GT dir structure
	ground_truth_dir_structure(root_dir, classes)

	# Intrinsic Parameters of the LineMOD Dataset Camera
	fx = 572.41140
	px = 325.26110
	fy = 573.57043
	py = 242.04899
	intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])



	print("------ Start creating ground truth ------")
	create_GT_masks(root_dir, background_dir, intrinsic_matrix, classes)
	create_UV_XYZ_dictionary(root_dir, classes)  # create UV - XYZ dictionaries
	print("----- Finished creating ground truth -----")
