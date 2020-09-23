import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os, tqdm
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import skimage
from lcfcn import lcfcn_loss
from src import models
from haven import haven_img as hi
from scipy import ndimage
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import cv2
from haven import haven_img
from haven import haven_utils as hu
from src.models import base_networks, metrics
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


class PONet(torch.nn.Module):
    def __init__(self, exp_dict, train_set):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = 2
        self.exp_dict = exp_dict
        
        self.model_base = smp.FPN('resnet50',classes=self.n_classes,encoder_weights = None, decoder_merge_policy = 'cat')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model_base = nn.DataParallel(self.model_base)

        self.model_base.to(self.device)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=self.exp_dict["lr"], betas=(0.99, 0.999), weight_decay=0.0005)

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=self.exp_dict["lr"])

        else:
            raise ValueError
        
        self.scheduler = ReduceLROnPlateau(self.opt,  mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=5e-7, eps=1e-08)

    def train_on_loader(self, train_loader):
        self.train()
        n_batches = len(train_loader)
        train_meter = metrics.Meter()
        
        pbar = tqdm.tqdm(total=n_batches)
        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_meter.add(score_dict['train_loss'], 1)

            pbar.set_description("Training Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        self.scheduler.step(train_meter.get_avg_score())
        pbar.close()
        
        return {'train_loss':train_meter.get_avg_score()}

    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir_images=None, n_images = 3):
        self.eval()

        n_batches = len(val_loader)
        val_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(tqdm.tqdm(val_loader)):
            score_dict = self.val_on_batch(batch)
            val_meter.add(score_dict['valloss'], batch['images'].shape[0])
            
            pbar.update(1)

            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok=True)
                self.vis_on_batch(batch, savedir_image=os.path.join(
                    savedir_images, "%d.jpg" % i))
                
                pbar.set_description("Validating. MAE: %.4f" % val_meter.get_avg_score())

        pbar.close()
        val_mae = val_meter.get_avg_score()
        val_dict = {'val_mae':val_mae, 'val_score':val_mae}
        return val_dict
    
    @torch.no_grad()
    def test_on_loader(self, test_loader):
        self.eval()

        n_batches = len(test_loader)
        test_meter = metrics.Meter()
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(tqdm.tqdm(test_loader)):
            score_dict = self.test_on_batch(batch)
            test_meter.add(score_dict['testloss'], batch['images'].shape[0])
            
            pbar.update(1)
                
            pbar.set_description("Testing. iou: %.4f" % test_meter.get_avg_score())

        pbar.close()
        test_iou = test_meter.get_avg_score()
        test_dict = {'test_iou':test_iou, 'test_score':-test_iou}
        return test_dict
    
    
    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images = batch["images"].to(self.device)
        points = batch["points"].long().to(self.device)
        bkgs = batch["bkg"].long().to(self.device)
        objs = batch["obj"].to(self.device)
        masks = batch["gt"].to(self.device)
        logits = self.model_base.forward(images)

#         import pdb
#         pdb.set_trace()
        loss = lcfcn_loss.compute_obj_loss(logits, objs)+0.5*lcfcn_loss.compute_weighted_crossentropy(logits, points, bkgs)

        loss.backward()

        self.opt.step()

        return {"train_loss":loss.item()}


    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt":self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    def val_on_batch(self, batch):
        self.eval()
        images = batch["images"].to(self.device)
        mask = batch["gt"].to(self.device)
        
        logits = self.model_base.forward(images)
        
        prob = logits.sigmoid()
        val_loss = self.iou_pytorch(prob[:,1], mask)

        return {'valloss': val_loss.item()}
        
    def test_on_batch(self, batch):
        self.eval()
        images = batch["images"].to(self.device)
        mask = batch["gt"].to(self.device)
        logits = self.model_base.forward(images)
        prob = logits.sigmoid()
        test_loss = self.iou_pytorch(prob[:,1], mask)
        return {"testloss": test_loss.item()}
        
        
        
    
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        images = batch["images"].to(self.device)
        mask = batch["gt"].to(self.device)
        logits = self.model_base.forward(images)
        prob = logits.sigmoid()
        fig,axes = plt.subplots(1,3,figsize = (30,10))
        axes[0].imshow(images[0].detach().cpu().numpy().transpose(1,2,0))
        axes[1].imshow(mask[0].detach().cpu().numpy())
        axes[2].imshow(prob[0,1].detach().cpu().numpy()>0.5)
        for ax in axes:
            ax.axis('off')
        fig.savefig(savedir_image)
        plt.close()
        
    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        SMOOTH = 1e-6
        outputs = outputs.squeeze(1).round().bool() if not outputs.dtype is torch.bool else outputs
        labels = labels.squeeze(1).round().bool() if not labels.dtype is torch.bool else labels
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        return torch.mean(iou)