# import h5py
# from albumentations import Compose, HueSaturationValue
from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu


class HEDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, option="Train", random_seed=123, n_classes=1, augmul=500):

        self.transform = transform
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.option = option
        self.files_no = len(glob.glob(os.path.join(self.data_dir, option, "Norms", "*.png")))

        if self.transform:
            self.augmul = augmul
            np.random.seed(random_seed)
            self.random_seeds = np.random.randint(0, self.augmul*self.files_no*100,
                                                  (self.augmul*self.files_no,))

    def __getitem__(self, ind):
        real_ind = ind % self.files_no + 1
        image = sio.imread(os.path.join(self.data_dir, self.option, "Norms", self.option.lower()+'_'+str(real_ind)+".png"))[..., :3]
        obj = sio.imread(os.path.join(self.data_dir, self.option, "Objs", self.option.lower()+'_'+str(real_ind)+".tif"))
        bkg = sio.imread(os.path.join(self.data_dir, self.option, "Bkgs", self.option.lower()+'_'+str(real_ind)+".tif"))
        mask = sio.imread(os.path.join(self.data_dir, self.option, "GTs", self.option.lower()+'_'+str(real_ind)+".tif"))
        region = sio.imread(os.path.join(self.data_dir, self.option, "Regions", self.option.lower()+'_'+str(real_ind)+".tif"))
        points = hu.load_json(os.path.join(self.data_dir, self.option, "Pts", self.option.lower()+'_'+str(real_ind)+".json"))

        if self.transform:
            random_seed = self.random_seeds[ind]
            random.seed(random_seed)
            transformed = self.transform(image=image,
                                         keypoints=points["1"],
                                         keypoints0=points["2"],
                                         keypoints1=points["3"],
                                         keypoints2=points["4"],
                                         keypoints3=points["5"],
                                         keypoints4=points["6"],
                                         keypoints5=points["7"],
                                         mask=mask,
                                         mask0=bkg,
                                         mask1=obj,
                                         mask2=region)
            image = transformed["image"]
            points["1"] = np.array(transformed["keypoints"]).astype(int)
            points["2"] = np.array(transformed["keypoints0"]).astype(int)
            points["3"] = np.array(transformed["keypoints1"]).astype(int)
            points["4"] = np.array(transformed["keypoints2"]).astype(int)
            points["5"] = np.array(transformed["keypoints3"]).astype(int)
            points["6"] = np.array(transformed["keypoints4"]).astype(int)
            points["7"] = np.array(transformed["keypoints5"]).astype(int)
            mask = transformed["mask"]
            bkg = transformed["mask0"]
            obj = transformed["mask1"]
            region = transformed["mask2"]
            
            point_label = np.zeros_like(mask)
            counts = 0
            for k, v in points.items():
                counts += len(v)
                if len(v) > 0:
                    point_label[v[:, 1], v[:, 0]] = int(k)
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                    'points': torch.FloatTensor(point_label),
                    'bkg': torch.FloatTensor(bkg),
                    'obj': torch.FloatTensor(obj),
                    'gt': torch.FloatTensor(mask),
                    'region': torch.FloatTensor(region),
                    'counts': counts,
                    'meta': {'index': ind}}
        else:
            counts = len(points)
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1))/255.0,
                    'counts': counts,
                    'meta': {'index': ind},
                    'gt': torch.FloatTensor(mask)}
    
    def __len__(self):
        if self.option is "Train":
            return self.augmul*self.files_no
        else:
            return self.files_no
