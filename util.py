import jittor as jt
from jittor import Module

import math

import cv2

import numpy as np

# loss function
class CharbonnierLoss(Module):
    """L1 Charbonnier loss."""
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def execute(self, X, Y):
        diff = jt.add(X, -Y)
        error = jt.sqrt( diff * diff + self.eps )
        loss = jt.sum(error) 
        return loss

# decay helper for learning rate 
class LRScheduler:
    def __init__(self, optimizer, base_lr):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.lr_decay = 0.1
        self.decay_step = 100

    def step(self, epoch):
        self.optimizer.lr = self.base_lr * (self.lr_decay ** (epoch // self.decay_step))

# compute peak SNR
def compute_psnr(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

# resizing functions
# generate x2
def scale_half(img):
    resized = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), cv2.INTER_CUBIC)
    return resized

# generate low res
def scale_lr(img):
    resized = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4), cv2.INTER_CUBIC)
    return resized

# check if a file is image
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])