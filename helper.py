import sys
import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F


# Pickle functions to save and load dictionaries
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# helper function to plot grpahs

def showImage(window="image", img=None, hold=True):
    cv2.imshow(window, img)
    if hold:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            sys.exit(0)


def visualize(array):
    "Plot all images in the array of tensors in one row"
    for z in range(0, len(array)):
        temp = array[z]
        if temp.ndim > 3:  # tensor output in the form NCHW
            temp = (torch.argmax(temp, dim=1).squeeze())
        if len(temp.shape) >= 3:
            plt.figure()
            plt.imshow(np.transpose(
                temp.detach().numpy().squeeze(), (1, 2, 0)))
            plt.show()
        else:
            plt.figure()
            plt.imshow(temp.detach().numpy(), cmap='gray')


def create_bounding_box(img, pose, pt_cld_data, intrinsic_matrix,color=(0,0,255)):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data
    min_x, min_y, min_z = pt_cld_data.min(axis=0)
    max_x, max_y, max_z = pt_cld_data.max(axis=0)
    corners_3D = np.array([[max_x, min_y, min_z],
                           [max_x, min_y, max_z],
                           [min_x, min_y, max_z],
                           [min_x, min_y, min_z],
                           [max_x, max_y, min_z],
                           [max_x, max_y, max_z],
                           [min_x, max_y, max_z],
                           [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), color, 3)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), color, 3)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), color, 3)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), color, 3)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), color, 3)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), color, 3)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), color, 3)
    img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), color, 3)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), color, 3)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), color, 3)
    img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), color, 3)
    img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), color, 3)

    return img


def color_linemod_idmask_img(img):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors_LUT = np.array([[[0,0,0] for i in range(256)]],dtype=np.uint8)

    colors_LUT[0][1] = (0,0,90)        # ape
    colors_LUT[0][2] = (215,0,0)       # benchviseblue
    colors_LUT[0][3] = (155,155,155)   # cam
    colors_LUT[0][4] = (200,200,200)   # watering can
    colors_LUT[0][5] = (215,215,255)   # cat
    colors_LUT[0][6] = (0,90,0)        # driller
    colors_LUT[0][7] = (0,255,255)     # duck
    colors_LUT[0][8] = (90,90,90)      # eggbox
    colors_LUT[0][9] = (227,162,184)   # glue
    colors_LUT[0][10] = (90,0,0)       # holepuncher
    colors_LUT[0][11] = (255,255,175)  # iron
    colors_LUT[0][12] = (255,255,255)  # lamp
    colors_LUT[0][13] = (50,30,30)     # phone

    return cv2.LUT(color_img, colors_LUT)


# Using BGR format and encoding U to G and V to B
def color_uv(umask, vmask):
    assert(umask.shape == vmask.shape)
    #ulut = np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)
    #vlut = np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)

    # Initialize color_img (G channel to U values)
    color_img = cv2.cvtColor(umask, cv2.COLOR_GRAY2BGR)
    # B channel to V values
    color_img[:,:,0] = vmask
    # Zero out R channel
    color_img[:,:,2] = 0

    return color_img

# Expecting raw logits tensor with 14 channels
def show_predictions_tiled(pred, hold=True):
    #if pred.dtype != np.uint8 or np.max(pred) > 255 or np.min(pred) < 0:
    #    pred = cv2.normalize(src=pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    pred = np.squeeze(pred)
    # Apply softmax to get probabilities for each class
    probs = F.softmax(pred, dim=0)
    # Normalize for viewing
    probs = probs.cpu().detach().numpy()
    probs = cv2.normalize(src=probs, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Stack for viewing
    top = np.hstack(probs[0:7,:,:])
    bot = np.hstack(probs[7:14,:,:])
    pred_tiled = np.vstack((top,bot))
    if hold:
        showImage("Tiled predictions", pred_tiled)
    else:
        cv2.imshow("Tiled predictions", pred_tiled)

def draw_axis(img, pose, intrinsic_matrix, colors=[(0,0,255), (0,255,0), (255,0,0)], axis_len=10):
    # If only one color is specified
    if np.shape(colors) != (3,3):
        colors = [colors] * 3

    rvec = pose[0:3, 0:3]
    tvec = pose[:,3]

    origin_pt_3d = np.array([0,0,0], dtype=np.float32)
    origin_pt, jac = cv2.projectPoints(origin_pt_3d, rvec, tvec, intrinsic_matrix, None)
    origin_pt = np.int32(np.squeeze(origin_pt))

    axis = np.float32([[axis_len,0,0], [0,axis_len,0], [0,0,axis_len]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, intrinsic_matrix, None)

    # Use same color-axis convention as RVIZ, X-R, Y-G, Z-B
    img = cv2.line(img, tuple(origin_pt), tuple(imgpts[0].ravel()), colors[0], 2) # X - R
    img = cv2.line(img, tuple(origin_pt), tuple(imgpts[1].ravel()), colors[1], 2) # Y - G
    img = cv2.line(img, tuple(origin_pt), tuple(imgpts[2].ravel()), colors[2], 2) # Z - B

    return img


#def ADD_score(pt_cld, true_pose, pred_pose, diameter):
def draw_point_cloud(img, pose, pt_cld, intrinsic_matrix, color=(0,0,255)):
    pts_3d = np.float32(pt_cld[::10]) # Add some sparsity
    pts, jac = cv2.projectPoints(pts_3d, pose[0:3, 0:3], pose[:,3], intrinsic_matrix, None)
    pts = np.int32(np.squeeze(pts))
    for pt in pts:
        img = cv2.circle(img, (pt[0],pt[1]), 1, (0,0,255), 1)
    return img

#Evaluation metric - ADD score
def ADD_vis(img, pt_cld, true_pose, pred_pose, intrinsic_matrix):

    # Add column of ones to pt_cld for homog transform
    ones = np.ones((pt_cld.shape[0], 1))
    pt_cld = np.hstack((pt_cld, ones))

    target_pt_cld = pt_cld @ true_pose.T
    output_pt_cld = pt_cld @ pred_pose.T
    assert(len(target_pt_cld) == len(output_pt_cld))

    zero_rot = np.eye(3)
    zero_trans = np.float32([0,0,0])
    target_pts, jac = cv2.projectPoints(target_pt_cld, zero_rot, zero_trans, intrinsic_matrix, None)
    output_pts, jac = cv2.projectPoints(output_pt_cld, zero_rot, zero_trans, intrinsic_matrix, None)

    target_pts = np.int32(np.squeeze(target_pts))
    for pt in target_pts[::10]:
        img = cv2.circle(img, (pt[0],pt[1]), 1, (0,255,0), 1)

    output_pts = np.int32(np.squeeze(output_pts))
    for pt in output_pts[::10]:
        img = cv2.circle(img, (pt[0],pt[1]), 1, (0,0,255), 1)

    # TODO - draw a sparser cloud, and connect each point correspondence
    avg_distance = (np.linalg.norm(output_pt_cld - target_pt_cld)) / pt_cld.shape[0]
    avg_distance = np.sum(np.linalg.norm(output_pt_cld - target_pt_cld, axis=1)) / pt_cld.shape[0]
    #print("avg_distance %f: " % avg_distance)

    return img


#Evaluation metric - ADD score
def ADD_score(pt_cld, true_pose, pred_pose, diameter_threshold):
    # Shouldn't be doing the following 2 lines:
    #pred_pose[0:3, 0:3][np.isnan(pred_pose[0:3, 0:3])] = 1
    #pred_pose[:, 3][np.isnan(pred_pose[:, 3])] = 0

    target = pt_cld @ true_pose[0:3, 0:3] + np.array([true_pose[0, 3], true_pose[1, 3], true_pose[2, 3]])
    output = pt_cld @ pred_pose[0:3, 0:3] + np.array([pred_pose[0, 3], pred_pose[1, 3], pred_pose[2, 3]])

    # This was a bug in the original contributor's implementation. This gives ~56% accuracy, where as
    # actual results are no where near that. Running the correspondence block only gives 0% accuracy.
    #avg_distance = (np.linalg.norm(output - target)) / pt_cld.shape[0]
    avg_distance = np.sum(np.linalg.norm(output - target, axis=1)) / pt_cld.shape[0]

    if avg_distance <= diameter_threshold:
        return 1, avg_distance
    else:
        return 0, avg_distance
