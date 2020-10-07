from haven import haven_chk as hc
# from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from albumentations import Flip, ShiftScaleRotate, RandomRotate90, RandomCrop
import albumentations as A
from src.datasets import HEDataset
import glob,os

cudnn.benchmark = True

def loadsplittedfolds(fold_folder_path):
    return np.loadtxt(os.path.join(fold_folder_path,'tr_fold.txt')).astype(int).tolist(),np.loadtxt(os.path.join(fold_folder_path,'val_fold.txt')).astype(int).tolist(),np.loadtxt(os.path.join(fold_folder_path,'test_fold.txt')).astype(int).tolist()

def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0):
    # bookkeepting stuff
    # ==================
    
#     tr_idx, val_idx, ts_idx = loadsplittedfolds(folddir)
    
    savedir = os.path.join(savedir_base, exp_dict["model"]["name"])
    os.makedirs(savedir, exist_ok=True)
    
    if reset:
        hc.delete_and_backup_experiment(savedir)

    print("Experiment saved in %s" % savedir)

    # Dataset
    # ==================
    # train set
    
#     data_transform = A.Compose([A.RandomCrop(64,64), 
#                                 A.Flip(p=0.5)], 
#                                keypoint_params = A.KeypointParams(format='xy'), 
#                                additional_targets={'bkg':'mask', 
#                                                    'obj':'mask'})
    data_transform = A.Compose([A.RandomCrop(128,128), 
                           A.Flip(), 
                           A.IAAAffine(), 
                           A.Rotate(), 
                           A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit=15, val_shift_limit=10,p=0.3), 
                           A.GaussianBlur(3, p=0.3), 
                           A.GaussNoise(30, p=0.3)], 
                          keypoint_params = A.KeypointParams(format='xy'), 
                          additional_targets = {'mask0':'mask', 
                                                'mask1':'mask',
                                                'mask2':'mask',
                                                'keypoints0':'keypoints', 
                                                'keypoints1':'keypoints', 
                                                'keypoints2':'keypoints', 
                                                'keypoints3':'keypoints', 
                                                'keypoints4':'keypoints', 
                                                'keypoints5':'keypoints'})
    
#     data_transform = [
#         Flip(p=0.5),
#     #     ShiftScaleRotate(p=0.5),
#         RandomRotate90(p=0.5),
#         RandomCrop(256,256)
#     ]
    train_set = HEDataset(data_dir = datadir, 
                          transform=data_transform, 
                          option = "Train")

    
    test_transform = A.Compose([A.Resize(1024,1024)], 
                              keypoint_params = A.KeypointParams(format='xy'), 
                              additional_targets = {'mask0':'mask', 
                                                    'mask1':'mask'})
    # val set
    val_set = HEDataset(data_dir = datadir,
                        transform = test_transform, 
                        option = "Validation")
    

#     val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
#                             sampler=val_sampler,
                            batch_size=1,
                            num_workers=num_workers)
    
    # test set
    test_set = HEDataset(data_dir = datadir, 
                         transform = test_transform,
                         option = "Test")
    

#     val_sampler = torch.utils.data.SequentialSampler(val_set)
    test_loader = DataLoader(test_set,
#                             sampler=val_sampler,
                            batch_size=1,
                            num_workers=num_workers)
    # Model
    # ==================
    
    model = models.get_model(exp_dict['model'],exp_dict = exp_dict,train_set = train_set).cuda()

    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % (s_epoch))
    
#     train_sampler = torch.utils.data.RandomSampler(
#         train_set, replacement=True, num_samples=2*len(val_set))

    train_loader = DataLoader(train_set,
                              batch_size=exp_dict["batch_size"], 
                              num_workers=num_workers)

    for e in range(s_epoch, exp_dict['max_epoch']):
        # Validate only at the start of each cycle
        score_dict = {}

        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate and Visualize the model
        val_dict = model.val_on_loader(val_loader, 
                        savedir_images=os.path.join(savedir, "images"),
                        n_images=7)
        score_dict.update(val_dict)

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = len(score_list)

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

        # Save Best Checkpoint
        if e == 0 or (score_dict.get("val_score", 0) > score_df["val_score"][:-1].fillna(0).max()):
            hu.save_pkl(os.path.join(
                savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                          model.get_state_dict())
            print("Saved Best: %s" % savedir)
    
    test_dict = model.test_on_loader(test_loader)
    hu.save_pkl(os.path.join(savedir, 'test_iou.pkl'),test_dict)
    print('Test IoU:{}'.format(test_dict["test_iou"]))
    print('Experiment completed et epoch %d' % e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_dict', required=True, type=str)
#     parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
#     parser.add_argument("-ei", "--exp_id", default=None)
#     parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Collect experiments
    # ===================
    
    exp_dict = hu.load_json(args.exp_dict)
    # do trainval
    folddir_10 = glob.glob(os.path.join(args.datadir, '*_fold'))

    for folddir in folddir_10:
        savedir_base = os.path.join(folddir,'Result')
        os.makedirs(savedir_base,exist_ok=True)
#         trainval(exp_dict, savedir_base, datadir, folddir, reset=True, num_workers=25)
        trainval(exp_dict=exp_dict,
                savedir_base=savedir_base,
                datadir=args.datadir,
                folddir = folddir,
                reset=args.reset,
                num_workers=args.num_workers)
