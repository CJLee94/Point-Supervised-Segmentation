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


def get_model_base(model_base_dict, in_channels=3, n_classes=2):
    if model_base_dict['pretrained']:
        encoder_weights = 'imagenet'
    else:
        encoder_weights = None
    if model_base_dict['name'].lower() == 'fpn':
        return smp.FPN(
            'resnet50',
            in_channels=in_channels,
            classes=n_classes,
            encoder_weights=encoder_weights,
            decoder_merge_policy='cat'
        )
    elif model_base_dict['name'].lower() == 'fpn_half':
        return smp.FPN(
            'resnet50',
            encoder_depth=3,
            in_channels=in_channels,
            classes=n_classes,
            encoder_weights=encoder_weights,
            decoder_pyramid_channels=64,
            decoder_segmentation_channels=32,
            upsampling=1,
            decoder_merge_policy='cat'
        )
    elif model_base_dict['name'].lower() == 'unet':
        return smp.Unet(
            'densenet121',
            in_channels=in_channels,
            classes=n_classes,
            encoder_weights=encoder_weights
        )
    elif model_base_dict['name'].lower() == 'unet_half':
        return smp.Unet(
            'densenet121',
            encoder_depth=3,
            in_channels=in_channels,
            classes=n_classes,
            encoder_weights=encoder_weights
        )
    else:
        raise TypeError('The network of your choice is not available currently, Sorry!')

class PseudoEdgeNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(PseudoEdgeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, n_classes, 3, padding=1)
        
    def __call__(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        return x

    
class PENet(torch.nn.Module):
    def __init__(self, exp_dict, train_set):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = train_set.n_classes
        self.exp_dict = exp_dict
        
        self.model_base = smp.FPN('resnet50', classes=self.n_classes, encoder_weights=None, decoder_merge_policy='cat')

        # PseudoEdge Net
        self.g = PseudoEdgeNet(2)
        
        # Attention Module
        self.h = smp.FPN('resnet18', classes=1, encoder_weights=None, decoder_merge_policy='cat', activation='sigmoid')
        
        # Sobel Filter
        S_weight=np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])
        S_weight = S_weight[:, np.newaxis, :, :]
        self.Sobel = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.Sobel.weight = nn.Parameter(torch.from_numpy(S_weight).float())
        self.Sobel.weight.requires_grad = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model_base = nn.DataParallel(self.model_base)
            self.h = nn.DataParallel(self.h)
            self.g = nn.DataParallel(self.g)
            self.Sobel = nn.DataParallel(self.Sobel)

        self.model_base.to(self.device)
        self.h.to(self.device)
        self.g.to(self.device)
        self.Sobel.to(self.device)

        self.bkg_enable = exp_dict["bkg_enable"]
        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam([{'params': self.model_base.parameters()},
                                         {'params': self.h.parameters()},
                                         {'params': self.g.parameters()}],
                                        lr=self.exp_dict["lr"],
                                        betas=(0.99, 0.999),
                                        weight_decay=0.0005)

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD([{"params": self.model_base.parameters()},
                                        {"params": self.h.parameters()},
                                        {"params": self.g.parameters()}],
                                       lr=self.exp_dict["lr"])

        else:
            raise ValueError
        
        self.scheduler = ReduceLROnPlateau(self.opt,  mode='min', factor=0.5,
                                           patience=5, verbose=True, threshold=0.0001,
                                           threshold_mode='rel', cooldown=0, min_lr=5e-7, eps=1e-08)

    def train_on_loader(self, train_loader):
        self.train()
        n_batches = len(train_loader)
        train_meter = metrics.Meter()
        
        pbar = tqdm.tqdm(total=n_batches)
        for batch in train_loader:
            score_dict = self.train_on_batch(batch)
            train_meter.add(score_dict['train_loss'], batch['images'].shape[0])

            pbar.set_description("Training Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        self.scheduler.step(train_meter.get_avg_score())
        pbar.close()
        
        return {'train_loss': train_meter.get_avg_score()}

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
        val_dict = {'val_mae': val_mae, 'val_score': val_mae}
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
        logits = self.model_base.forward(images)
        gI = self.g(images)
        hI = self.h(images)
        hIgI = gI * hI
        sfI = self.Sobel(logits[:, :1].sigmoid())

        loss = torch.mean(torch.abs(sfI-hIgI)) + \
               lcfcn_loss.compute_weighted_crossentropy(logits, points, bkgs,
                                                        weights=[1, 1],
                                                        bkg_enable=self.bkg_enable)

        loss.backward()

        self.opt.step()

        return {"train_loss": loss.item()}

    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    def val_on_batch(self, batch):
        self.eval()
        images = batch["images"].to(self.device)
        mask = batch["gt"].to(self.device)
        logits = self.model_base.forward(images)
        
        val_loss = self.iou_pytorch(logits.sigmoid(), mask)

        return {'valloss': val_loss.item()}
        
    def test_on_batch(self, batch):
        self.eval()
        images = batch["images"].to(self.device)
        mask = batch["gt"].to(self.device)
        logits = self.model_base.forward(images)
        prob = logits.sigmoid()
        test_loss = self.iou_pytorch(prob, mask)
        return {"testloss": test_loss.item()}
    
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        images = batch["images"].to(self.device)
        mask = batch["gt"].to(self.device)
        logits = self.model_base.forward(images)
        prob = logits.sigmoid()
        seg = torch.argmax(prob, dim=1)
        #         import pdb
        #         pdb.set_trace()
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        axes[0].imshow(images[0].detach().cpu().numpy().transpose(1, 2, 0))
        axes[1].imshow(mask[0].detach().cpu().numpy(), vmax=7, vmin=0)
        axes[2].imshow(seg[0].detach().cpu().numpy(), vmax=7, vmin=0)
        for ax in axes:
            ax.axis('off')
        fig.savefig(savedir_image)
        plt.close()
        
    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        smooth = 1e-6
        outputs = torch.argmax(outputs, dim=1) if outputs.dtype is not torch.bool else outputs
        labels = labels.squeeze(1).round() if labels.dtype is not torch.bool else labels
        iou = 0.0
        for cls in range(8):
            outputs_cls = outputs == cls
            labels_cls = labels == cls
            intersection = (outputs_cls & labels_cls).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
            union = (outputs_cls | labels_cls).float().sum((1, 2))  # Will be zzero if both are 0

            iou += (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0

        iou /= 8
        return torch.mean(iou)
