# import h5py
# from albumentations import Compose, HueSaturationValue
from torch.utils.data import Dataset
import numpy as np
import torch
import glob,os
import skimage.io as sio
from haven import haven_utils as hu

class HEDataset(Dataset):
    
    def __init__(self, data_dir, transform=None, option="Train", n_classes = 1):

        self.transform = transform
        self.datafp = datapath
        self.n_classes = n_classes
        self.names = glob.glob(os.path.join(data_dir,'**',option, "*_norm.tif"))
        self.names = [path[:-10] for path in self.names]
#         #names = list(map(str,list(np.sort(list(map(int,self.data.keys()))))))
#         with h5py.File(self.datafp,'r') as h5f:
#             names = np.sort(list(h5f.keys())).tolist()
#             if index is None:
#                 self.names = names
#             else:
#                 self.names = (np.array(names)[index]).tolist()
        
    def __getitem__(self,ind):
#         with h5py.File(self.datafp,'r') as h5f:
#             ind = ind%len(self.names)
#             All = h5f[self.names[ind]]
#             X = All[:3].transpose(1,2,0)
#             point_label = All[3]
#             bkg = All[4]
#             mask = All[5]
        real_ind = ind%(len(self.names))
        image = sio.imread(self.names[real_ind]+"_norm.tif")
        obj = sio.imread(self.names[real_ind]+"_obj.tif")
        bkg = sio.imread(self.names[real_ind]+"_bkg.tif")
        points = hu.load_json(self.names[real_ind]+"_points.json")

        if self.transform:
#             color_aug = HueSaturationValue(hue_shift_limit = 10, sat_shift_limit=15, val_shift_limit=10,p=0.3)
#             target = {}
#             target['image'] = 'image'
#             target['image0'] = 'image'
#             target['image1'] = 'image'
#             target['image2'] = 'image'
#             transformed = Compose(self.transform,p=1,additional_targets=target)(image=X,image0=point_label,image1 = bkg, image2 = mask)
            transformed = self.transform(image = image, keypoints = points, mask = mask, mask0 = bkg, mask1 = obj)
#             point_label = transformed['image0']
#             point_label = point_label.clip(0,1).round().astype('uint8')
#             bkg = transformed['image1']
#             bkg = bkg.clip(0,1).round().astype('uint8')

#             X = (transformed['image']*255).astype(np.uint8)
#             X = color_aug(image = X)['image'].transpose(2,0,1)/255

            image = transformed["image"]
            points = transformed["keypoints"]
            mask = transformed["mask"]
            bkg = transformed["mask0"]
            obj = transformed["mask1"]
            counts = torch.LongTensor(len(points))
            
            point_label = np.zeros_like(mask)
            point_label[points[:,1], points[:,0]] = 1
            return {'images':torch.FloatTensor(image.transpose(1,2,0)),
                    'points':torch.FloatTensor(point_label),
                    'bkg': torch.FloatTensor(bkg),
                    'obj': torch.FloatTensor(obj),
                    'counts': counts,
                    'meta':{'index':ind}}
        else:
            counts = torch.LongTensor(np.array([int(point_label.sum())]))
            return {'images':torch.FloatTensor(X.transpose(2,0,1)),
#                     'points':torch.FloatTensor(point_label.squeeze()),
#                     'bkg': torch.FloatTensor(bkg.squeeze()),
                    'counts': counts,
                    'meta':{'index':ind},
                    'gt': mask}
    
    def __len__(self):
        if not self.transform is None:
            return 100*len(self.names)
        else:
            return len(self.names)
7