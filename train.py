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

from helper import save_obj
from pose_block import initial_pose_estimation
from create_renderings import create_refinement_inputs
from pose_refinement import train_pose_refinement
from correspondence_block import train_correspondence_block
from create_ground_truth import create_GT_masks, create_UV_XYZ_dictionary, check_dataset_dir_structure

# Update classes if you're using a different dataset
classes = {'ape': 1, 'benchviseblue': 2, 'cam': 3, 'can': 4, 'cat': 5, 'driller': 6,
			'duck': 7, 'eggbox': 8, 'glue': 9, 'holepuncher': 10, 'iron': 11, 'lamp': 12, 'phone': 13}

list_of_actions = ["train_correspondence", "initial_pose_estimation", "create_refinement_inputs", "train_pose_refinement"]

parser = argparse.ArgumentParser(
    description='Script to create the Ground Truth masks')
parser.add_argument("root_dir", help="path to dataset directory (LineMOD_Dataset)")
parser.add_argument("train_eval_dir", help="path to dir to store training run specific info")
parser.add_argument("action", help="path to dir to store training run specific info", choices=list_of_actions)
parser.add_argument("--epochs", default=20, type=int, help="correspondence block epochs")
parser.add_argument("--batch_size", default=4, type=int, help="batch size (default: 4)")
parser.add_argument("--corr_block_out", help="correspondence block output dir AND filename")
args = parser.parse_args()

root_dir = args.root_dir
train_eval_dir = args.train_eval_dir
action = args.action
print("Action: %s" % action)

# Check we at least have the dirs we're expecting
check_dataset_dir_structure(root_dir, train_eval_dir, classes)

# Intrinsic Parameters of the Camera
fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])


if action == "train_correspondence":
	print("Training for %i epochs" % args.epochs)
	print("------ Started training of the correspondence block ------")
	train_correspondence_block(root_dir, train_eval_dir, classes, epochs=args.epochs, \
		batch_size=args.batch_size, out_path_and_name=args.corr_block_out)
	print("------ Training Finished ------")

if action == "initial_pose_estimation":
	print("------ Started Initial pose estimation ------")
	# This creates predicted pose images for all training data
	initial_pose_estimation(root_dir, train_eval_dir, classes, intrinsic_matrix)
	print("------ Finished Initial pose estimation -----")

if action == "create_refinement_inputs":
	print("----- Started creating inputs for DL based pose refinement ------")
	# This creates /pose_refinement/rendered/color /pose_refinement/real/color for all training data
	create_refinement_inputs(root_dir, train_eval_dir, classes, intrinsic_matrix)
	print("----- Finished creating inputs for DL based pose refinement")

if action == "train_pose_refinement":
	print("----- Started training DL based pose refiner ------")
	train_pose_refinement(root_dir, train_eval_dir, classes, epochs=10)
	print("----- Finished training DL based pose refiner ------")
