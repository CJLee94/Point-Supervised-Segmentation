import h5py
from albumentations import Compose, HueSaturationValue
from torch.utils.data import Dataset
import numpy as np
import torch

class HEDataset(Dataset):
    
    def __init__(self, datapath, transform=None, index = None):

        self.transform = transform
        self.datafp = datapath
        self.n_classes = 2
        #names = list(map(str,list(np.sort(list(map(int,self.data.keys()))))))
        with h5py.File(self.datafp,'r') as h5f:
            names = np.sort(list(h5f.keys())).tolist()
            if index is None:
                self.names = names
            else:
                self.names = (np.array(names)[index]).tolist()
        
    def __getitem__(self,ind):
        with h5py.File(self.datafp,'r') as h5f:
            ind = ind%len(self.names)
            All = h5f[self.names[ind]]
            X = All[:3].transpose(1,2,0)
            point_label = All[3]
            bkg = All[4]
            mask = All[5]

        if self.transform:
            color_aug = HueSaturationValue(hue_shift_limit = 10, sat_shift_limit=15, val_shift_limit=10,p=0.3)
            target = {}
            target['image'] = 'image'
            target['image0'] = 'image'
            target['image1'] = 'image'
            target['image2'] = 'image'
            transformed = Compose(self.transform,p=1,additional_targets=target)(image=X,image0=point_label,image1 = bkg, image2 = mask)
            point_label = transformed['image0']
            point_label = point_label.clip(0,1).round().astype('uint8')
            bkg = transformed['image1']
            bkg = bkg.clip(0,1).round().astype('uint8')

            X = (transformed['image']*255).astype(np.uint8)
            X = color_aug(image = X)['image'].transpose(2,0,1)/255
            counts = torch.LongTensor(np.array([int(point_label.sum())]))
            return {'images':torch.FloatTensor(X),
                    'points':torch.FloatTensor(point_label.squeeze()),
                    'bkg': torch.FloatTensor(bkg.squeeze()),
                    'counts': counts,
                    'meta':{'index':ind}}
        else:
            counts = torch.LongTensor(np.array([int(point_label.sum())]))
            return {'images':torch.FloatTensor(X.transpose(2,0,1)),
                    'points':torch.FloatTensor(point_label.squeeze()),
                    'bkg': torch.FloatTensor(bkg.squeeze()),
                    'counts': counts,
                    'meta':{'index':ind},
                    'gt': mask}
    
    def __len__(self):
        if not self.transform is None:
            return 100*len(self.names)
        else:
            return len(self.names)
