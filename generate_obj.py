import glob
import os
from haven import haven_utils as hu
import SimpleITK as sitk
import numpy as np
from skimage.segmentation import watershed, find_boundaries
from segmentation._watershed import watershed as watershed_p
import matplotlib.pyplot as plt
import skimage.io as sio

from skimage.segmentation import find_boundaries
from skimage.filters import gaussian, sobel, scharr, difference_of_gaussians
from segmentation._watershed import watershed

SMOOTH = 1e-6

edge_filter = sobel

data_dir = "./CoNSeP"

def validate_obj_params(obj_params=dict()):
    if not type(obj_params) is dict:
        raise TypeError('The obj_params is expected to be a dictionary not a {}'.format(type(obj_params)))
    if not 'sigma' in obj_params.keys():
        obj_params['sigma'] = 0.0
    if not 'alpha' in obj_params.keys():
        obj_params['alpha'] = 1.0
    return obj_params


def generate_dist(image, points, obj_params=dict()):
    obj_params = validate_obj_params(obj_params)
    points = np.array(points)
    pl = np.zeros(image.shape[:2])
    if len(points)!=0:
        pl[points[:, 1].round().astype(int), points[:, 0].round().astype(int)] = np.arange(1, len(points) + 1)
        image = gaussian(image, 2.1381697911810127, multichannel=True)
        #     image[...,0] = edge_filter(image[...,0])
        #     image[...,1] = edge_filter(image[...,1])
        #     image[...,2] = edge_filter(image[...,2])
        #     import pdb
        #     pdb.set_trace()
        return watershed(image, pl, method=True)
    else:
        return np.zeros(image.shape[:2]), np.inf*np.ones(image.shape[:2])


def generate_obj(image, points, obj_params=dict()):
    obj_params = validate_obj_params(obj_params)
    reg_split, dist = generate_dist(image, points, obj_params)
    bkg = find_boundaries(reg_split)
    #     dist[bkg] = np.inf
    return np.exp(-obj_params['alpha'] * dist ** 2), bkg, reg_split

def generate_obj_from_file(img_file, points_file, obj_params=dict()):
    obj_params = validate_obj_params(obj_params)
    img = sio.imread(img_file)
    points = hu.load_json(points_file)
    if type(points) is dict:
        dist = np.stack([generate_obj(img, v, obj_params)[0] for _, v in points.items()], axis=0)
        return dist
    elif type(points) is list:
        print("It's a List")
    else:
        print("Type Error:")

    return generate_obj(img, points, obj_params=obj_params)

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.round().astype(bool)
    labels = labels.round().astype(bool)
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou


def cross_entropy_numpy(outputs: np.array, labels: np.array):
    return np.mean(-labels * np.log(SMOOTH + outputs) - (1 - labels) * np.log(SMOOTH + 1 - outputs))


# DONE: Generate Voronoi background label
# save_dir = os.path.join(data_dir, "Train", "Bkgs_v")
# os.makedirs(save_dir, exist_ok=True)
# for train_pts in glob.glob(os.path.join(data_dir, "Train", "Pts", "*")):
#     points = hu.load_json(train_pts)
#     pl_arr = np.zeros((1000, 1000))
#     base_label = 1
#     for k, v in points.items():
#         if not len(v) == 0:
#             v = np.array(v).round().astype(int)
#             pl_arr[v[:, 1], v[:, 0]] = np.arange(base_label, len(v)+base_label)
#             base_label += len(v)
#
#     black = np.zeros_like(pl_arr)
#     labels = find_boundaries(watershed(black, pl_arr)).astype(float)
#     sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(labels), sitk.sitkFloat32),
#                     os.path.join(save_dir, os.path.split(train_pts)[-1][:-5]+".tif"))

# TODO: Test each preprocessing



# TODO: Generate class-wise proposed objectness

if __name__=="__main__":
    save_dir = os.path.join(data_dir, "Train", "Objs")
    os.makedirs(save_dir, exist_ok=True)
    # for train_pts in glob.glob(os.path.join(data_dir, "Train", "Pts", "*")):
    test_image = "/home/cjlee/Documents/lcfcn-pseudo/CoNSeP/Train/Norms/train_3.png"
    test_points = "/home/cjlee/Documents/lcfcn-pseudo/CoNSeP/Train/Pts/train_3.json"
    obj = generate_obj_from_file(test_image, test_points)

# TODO: Generate class-wise Gauss Region
