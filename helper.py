import sys
import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Pickle functions to save and load dictionaries
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# helper function to plot grpahs

def showImage(window="image", img=None):
    cv2.imshow(window, img)
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


def ADD_score(pt_cld, true_pose, pred_pose, diameter):
    "Evaluation metric - ADD score"
    #pred_pose[0:3, 0:3][np.isnan(pred_pose[0:3, 0:3])] = 1
    # The following is cheating...
    #pred_pose[:, 3][np.isnan(pred_pose[:, 3])] = 0
    target = pt_cld @ true_pose[0:3, 0:3] + np.array([true_pose[0, 3], true_pose[1, 3], true_pose[2, 3]])
    output = pt_cld @ pred_pose[0:3, 0:3] + np.array([pred_pose[0, 3], pred_pose[1, 3], pred_pose[2, 3]])
    avg_distance = (np.linalg.norm(output - target)) / pt_cld.shape[0]
    print("avg_distance %f: " % avg_distance)
    threshold = diameter * 0.1
    print("threshold %f: " % threshold)

    # Draw the pointcloud
    for pt in target:
        img = cv2.circle(full_img, (pt[0],pt[1]), 5, (0,0,255), 1)

    if avg_distance <= threshold:
        return 1
    else:
        return 0



