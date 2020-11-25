import sys
import os
import re
import cv2
import pickle
import random
import numpy as np
from helper import *

"""
Helper function to the read the rotation and translation file
    Args:
            rot_adr (str): path to the file containing rotation of an object
    tra_adr (str): path to the file containing translation of an object
    Returns:
            rigid transformation (np array): rotation and translation matrix combined
"""
def get_rot_tra(rot_adr, tra_adr):
    rot_matrix = np.loadtxt(rot_adr, skiprows=1)
    trans_matrix = np.loadtxt(tra_adr, skiprows=1)
    trans_matrix = np.reshape(trans_matrix, (3, 1))
    rigid_transformation = np.append(rot_matrix, trans_matrix, axis=1)

    return rigid_transformation

"""
Helper function to fill the holes in id , u and vmasks.
Note that this only works on masks with a single class id.
It'd be possible, but much more difficult to fill holes in multiple classes simultaneously.
This is not necessary for the LineMOD dataset.
Also note that further improvements could be made to fill in the 'bright' holes. These
are caused by the sparsity of the point cloud, where you can 'see through' the darker region.
Args:
    idmask (uint8 np.array): id mask whose holes you want to fill
    umask (uint8 np.array): u mask whose holes you want to fill
    vmask (uint8 np.array): v mask whose holes you want to fill
    class_id (uint8): class id of object
Returns:
    filled_id_mask (np array): id mask with holes filled
    filled_u_mask (np array): u mask with holes filled
    filled_id_mask (np array): v mask with holes filled
"""
def fill_holes(idmask, umask, vmask, class_id):
    #showImage("idmask", idmask*(255/np.max(idmask)))
    #showImage("umask", umask)
    #showImage("vmask", vmask)

    thr, im_th = cv2.threshold(idmask, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)

    # Inpainting won't work for class_id here, it'll just fill it with zeros since the source
    # image is also a mask
    filled_id_mask = idmask
    filled_id_mask[mask > 0] = class_id

    # Now we don't want to inpaint the whole mask region for U and V, but rather the gaps in it
    umask_gaps = np.uint8(umask == 0)
    #showImage("umask_gaps", cv2.resize(umask_gaps*255, (1600, 1200)))
    umask_gaps = cv2.bitwise_and(umask_gaps, umask_gaps, mask=mask)
    #showImage("umask_gaps2", cv2.resize(umask_gaps*255, (1600, 1200)))
    filled_u_mask = cv2.inpaint(umask, umask_gaps, 5, cv2.INPAINT_TELEA)
    # Alternately, we could median filter, this would also fill in the "bright spots"
    #filled_u_mask = cv2.medianBlur(umask, 5)

    vmask_gaps = np.uint8(vmask == 0)
    vmask_gaps = cv2.bitwise_and(vmask_gaps, vmask_gaps, mask=mask)
    filled_v_mask = cv2.inpaint(vmask, vmask_gaps, 5, cv2.INPAINT_TELEA)

    # Check that our filled mask only contains 0 and class_id
    assert((np.unique(filled_id_mask) == [0, class_id]).all())

    # Debugging - check the residuals of our operation to ensure we're not over-correcting
    #diff = np.uint8(abs(np.int16(filled_u_mask)-np.int16(umask)))
    #unique, counts = np.unique(diff, return_counts=True)
    #print(unique, "\n", counts)
    #showImage("diff u_mask", cv2.resize(diff, (1600, 1200)))

    #showImage("filled_id_mask", cv2.resize(idmask*(255/np.max(idmask)), (1600, 1200)))
    #showImage("filled_u_mask", cv2.resize(filled_u_mask, (1600, 1200)))
    #showImage("filled_v_mask", cv2.resize(filled_v_mask, (1600, 1200)))
    return filled_id_mask, filled_u_mask, filled_v_mask


def create_GT_masks(root_dir, background_dir, intrinsic_matrix, classes):
    """
    Helper function to create the Ground Truth ID,U and V masks
        Args:
        root_dir (str): path to the root directory of the dataset
        background_dir(str): path t
        intrinsic_matrix (array): matrix containing camera intrinsics
        classes (dict) : dictionary containing classes and their ids
        Saves the masks to their respective directories
    """
    list_all_images = load_obj(os.path.join(root_dir, "all_images_adr"))
    #training_images_idx = load_obj(root_dir + "train_images_indices")
    #for i in range(len(training_images_idx)):
    for i, img_adr in enumerate(list_all_images):
        # These are some bad ones for testing
        #img_adr = "/home/alex/rit/Research/Datasets/DPOD/LineMOD_Dataset2/glue/data/color1022.jpg"
        #img_adr = "/home/alex/rit/Research/Datasets/DPOD/LineMOD_Dataset2/glue/data/color444.jpg"
        #img_adr = list_all_images[training_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        regex = re.compile(r'\d+')
        idx = regex.findall(os.path.split(img_adr)[1])[0]

        if i % 1000 == 0:
            print(str(i) + "/" + str(len(list_all_images)) + " finished!")

        image = cv2.imread(img_adr)
        #print(img_adr)
        #showImage("image", cv2.resize(image, (1600, 1200)))

        ID_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        U_mask = np.zeros((image.shape[0], image.shape[1]))
        V_mask = np.zeros((image.shape[0], image.shape[1]))

        ID_mask_file = os.path.join(root_dir, label, "ground_truth/IDmasks/color" + str(idx) + ".png")
        U_mask_file = os.path.join(root_dir, label,  "ground_truth/Umasks/color" + str(idx) + ".png")
        V_mask_file = os.path.join(root_dir, label,  "ground_truth/Vmasks/color" + str(idx) + ".png")

        tra_adr = os.path.join(root_dir, label, "data/tra" + str(idx) + ".tra")
        rot_adr = os.path.join(root_dir, label, "data/rot" + str(idx) + ".rot")
        rigid_transformation = get_rot_tra(rot_adr, tra_adr)

        # Read point Point Cloud Data
        ptcld_file = os.path.join(root_dir, label, "object.xyz")
        pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        ones = np.ones((pt_cld_data.shape[0], 1))
        homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)

        # Perspective Projection to obtain 2D coordinates for masks
        homogenous_2D = intrinsic_matrix @ (rigid_transformation @ homogenous_coordinate.T)
        coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
        coord_2D = ((np.floor(coord_2D)).T).astype(int)
        x_2d = np.clip(coord_2D[:, 0], 0, 639)
        y_2d = np.clip(coord_2D[:, 1], 0, 479)
        ID_mask[y_2d, x_2d] = classes[label]

        # Generate Ground Truth UV Maps
        centre = np.mean(pt_cld_data, axis=0)
        length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                            pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
        unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                            1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
        U = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
        V = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
        U_mask[y_2d, x_2d] = U
        V_mask[y_2d, x_2d] = V
        # Convert float to uint8
        U_mask = cv2.normalize(src=U_mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        V_mask = cv2.normalize(src=V_mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Saving ID, U and V masks after using the fill holes function
        ID_mask, U_mask, V_mask = fill_holes(ID_mask, U_mask, V_mask, classes[label])
        cv2.imwrite(ID_mask_file, ID_mask)
        cv2.imwrite(U_mask_file, U_mask)
        cv2.imwrite(V_mask_file, V_mask)

        #if i % 100 != 0:  # change background for every 99/100 images
        # Generate the full set of background images so they're ready for use
        background_img_adr = os.path.join(background_dir, random.choice(os.listdir(background_dir)))
        background_img = cv2.imread(background_img_adr)
        background_img = cv2.resize(background_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        background_img[ID_mask>0] = image[ID_mask>0]
        background_adr = os.path.join(root_dir, label,  "changed_background/color" + str(idx) + ".png")
        cv2.imwrite(background_adr, background_img)


def create_UV_XYZ_dictionary(root_dir, classes):

    # create a dictionary for UV to XYZ correspondence
    for label in classes:
        ptcld_file = os.path.join(root_dir, label, "object.xyz")
        pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        # calculate u and v coordinates from the xyz point cloud file
        centre = np.mean(pt_cld_data, axis=0)
        length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                             pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
        unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                          1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
        u_coord = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
        v_coord = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
        u_coord = (u_coord * 255).astype(int)
        v_coord = (v_coord * 255).astype(int)
        # save the mapping as a pickle file
        dct = {}
        for u, v, xyz in zip(u_coord, v_coord, pt_cld_data):
            key = (u, v)
            if key not in dct:
                dct[key] = xyz
        save_obj(dct, os.path.join(root_dir, label, "UV-XYZ_mapping"))


def ground_truth_dir_structure(root_dir, classes):

    dirs = ["ground_truth", "ground_truth/IDmasks", "ground_truth/Umasks", \
            "ground_truth/Vmasks", "changed_background"]

    # create directories to store data
    for label in classes:
        for d in dirs:
            if not os.path.exists(os.path.join(root_dir, label, d)):
                os.mkdir(os.path.join(root_dir, label, d))


def test_dir_structure(test_val_dir, classes):

    dirs = ["predicted_pose", "pose_refinement", "pose_refinement/real", "pose_refinement/rendered"]

    for label in classes:  # create directories to store data
        for d in dirs:
            # Error out if dir already exists?
            if not os.path.exists(os.path.join(test_val_dir, label, d)):
                os.makedirs(os.path.join(test_val_dir, label, d))

def check_dataset_dir_structure(root_dir, test_val_dir, classes):

    gt_dirs = ["ground_truth", "ground_truth/IDmasks", "ground_truth/Umasks", \
            "ground_truth/Vmasks", "changed_background"]
    test_eval_dirs = ["predicted_pose", "pose_refinement", "pose_refinement/real", "pose_refinement/rendered"]

    for label in classes:
        for d in gt_dirs:
            if not os.path.exists(os.path.join(root_dir, label, d)):
                sys.exit("Error. '%s' doesn't exist" % os.path.join(root_dir, label, d))
        for d in test_eval_dirs:
            if not os.path.exists(os.path.join(test_val_dir, label, d)):
                sys.exit("Error. '%s' doesn't exist" % os.path.join(test_val_dir, label, d))
