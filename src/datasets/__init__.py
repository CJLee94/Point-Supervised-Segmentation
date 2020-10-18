# import h5py
# from albumentations import Compose, HueSaturationValue
from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
import random
from haven import haven_utils as hu
import SimpleITK as sitk


class HEDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=1, augmul=500, obj_option=None):

        self.transform = transform
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.option = option
        self.files_no = len(glob.glob(os.path.join(self.data_dir, option, "Norms", "*.png")))
        self.obj_option = obj_option

        if self.transform:
            self.augmul = augmul
            np.random.seed(random_seed)
            self.random_seeds = np.random.randint(0, self.augmul*self.files_no*100,
                                                  (self.augmul*self.files_no,))

    def __getitem__(self, ind):
        real_ind = ind % self.files_no + 1
        image = sio.imread(os.path.join(self.data_dir, self.option, "Norms", self.option.lower()+'_'+str(real_ind)+".png"))[..., :3]
        if self.obj_option == "Gauss":
            obj = sio.imread(os.path.join(self.data_dir, self.option, "GaussObj", self.option.lower() + '_' + str(real_ind) + ".tif"))
        else:
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


class HEDataset_Fast(Dataset):

    def __init__(self, data_dir, transform=None, option="Train",
                 random_seed=123, n_classes=1, augmul=500, patch_size=None, obj_option="Objs", bkg_option="Bkgs"):

        self.transform = transform
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.option = option
        self.files_no = len(glob.glob(os.path.join(self.data_dir, option, "Norms", "*.png")))
        self.obj_option = obj_option
        self.patch_size = patch_size
        self.bkg_option = bkg_option

        if self.transform:
            self.augmul = augmul
            np.random.seed(random_seed)
            self.random_seeds = np.random.randint(0, self.augmul * self.files_no * 100,
                                                  (self.augmul * self.files_no,))

    def __getitem__(self, ind):
        real_ind = ind % self.files_no + 1

        if self.transform:
            file_list = self.get_train_names(real_ind)
            image, obj, bkg, mask, region, points = self.random_read_subregion(file_list, random_seed=self.random_seeds[ind])

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
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0,
                    'points': torch.FloatTensor(point_label),
                    'bkg': torch.FloatTensor(bkg),
                    'obj': torch.FloatTensor(obj),
                    'gt': torch.FloatTensor(mask),
                    'region': torch.FloatTensor(region),
                    'counts': counts,
                    'meta': {'index': ind}}
        else:
            image = sio.imread(
                os.path.join(self.data_dir, self.option, "Norms", self.option.lower() + '_' + str(real_ind) + ".png"))[
                    ..., :3]
            mask = sio.imread(
                os.path.join(self.data_dir, self.option, "GTs", self.option.lower() + '_' + str(real_ind) + ".tif"))
            points = hu.load_json(
                os.path.join(self.data_dir, self.option, "Pts", self.option.lower() + '_' + str(real_ind) + ".json"))
            counts = len(points)
            return {'images': torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0,
                    'counts': counts,
                    'meta': {'index': ind},
                    'gt': torch.FloatTensor(mask)}

    def __len__(self):
        if self.option is "Train":
            return self.augmul * self.files_no
        else:
            return self.files_no

    def get_train_names(self, number):

        return os.path.join(self.data_dir, self.option, "Norms",
                            self.option.lower() + '_' + str(number) + ".png"), \
               os.path.join(self.data_dir, self.option, self.obj_option,
                            self.option.lower() + '_' + str(number) + ".tif"), \
               os.path.join(self.data_dir, self.option, self.bkg_option,
                            self.option.lower() + '_' + str(number) + ".tif"), \
               os.path.join(self.data_dir, self.option, "GTs",
                            self.option.lower() + '_' + str(number) + ".tif"), \
               os.path.join(self.data_dir, self.option, "Regions",
                            self.option.lower() + '_' + str(number) + ".tif"), \
               os.path.join(self.data_dir, self.option, "Pts",
                            self.option.lower() + '_' + str(number) + ".json")


    def random_read_subregion(self, file_list, random_seed=False):
        if random_seed:
            np.random.seed(random_seed)
        random_state = np.random.random(size=(2,))
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(file_list[0])
        file_reader.ReadImageInformation()
        image_size = file_reader.GetSize()
        extractindex = [int((img_dim-self.patch_size)*random_) for img_dim, random_ in zip(image_size, random_state)]

        file_reader.SetExtractIndex(extractindex)
        file_reader.SetExtractSize([self.patch_size, self.patch_size])

        return_item = [sitk.GetArrayFromImage(file_reader.Execute())[..., :3]]

        for file in file_list[1:-1]:
            file_reader.SetFileName(file)
            return_item.append(sitk.GetArrayFromImage(file_reader.Execute()))

        points_crop = dict()
        for k, v in hu.load_json(file_list[-1]).items():
            if len(v) == 0:
                points_crop[k] = v
            else:
                v = np.array(v)
                ind = np.logical_and(np.logical_and((v[:, 0]-extractindex[0]) >= 0,
                                                    (v[:, 0] < extractindex[0] + self.patch_size)),
                                     np.logical_and((v[:, 1]-extractindex[1]) >= 0,
                                                    (v[:, 1] < extractindex[1] + self.patch_size)))
                points_crop[k] = v[ind, :] - np.array(extractindex)[None]
        return_item.append(points_crop)
        return return_item

