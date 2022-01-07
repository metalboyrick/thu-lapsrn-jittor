import jittor as jt
import cv2
from jittor.dataset import Dataset
import os
import random
import numpy as np
from util import *

RANDOM_CROP_SIZE = 256

def random_crop(image, crop_height, crop_width):

    if image.shape[1] <= crop_width or image.shape[0] <= crop_height:
        image = cv2.resize(image, (image.shape[1] + crop_width, image.shape[0] + crop_height), cv2.INTER_CUBIC)

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    print(max_x, max_y)


    x = np.random.randint(0, max_x - 1)
    y = np.random.randint(0, max_y - 1)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


class LapSRNDataset(Dataset):
    def __init__(self, img_folder, rc_len=128,batch_size=64):
        super().__init__()
        self.image_filenames = [os.path.join(img_folder, x) for x in os.listdir(img_folder) if is_image_file(x)]
        
        self.tensor4x = np.zeros((len(self.image_filenames), 1, rc_len, rc_len))
        self.tensor2x = np.zeros((len(self.image_filenames), 1, rc_len // 2, rc_len // 2))
        self.tensorLR = np.zeros((len(self.image_filenames), 1, rc_len // 4, rc_len // 4))

        for index, filename in enumerate(self.image_filenames):
            
            _input = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            _input = random_crop(_input, RANDOM_CROP_SIZE, RANDOM_CROP_SIZE)      # random crop the input
            LR = scale_lr(_input)                       # generate lowres
            HR_2x = scale_half(_input)                  # generate 2x

            self.tensor4x[index, 0] = _input
            self.tensor2x[index, 0] = HR_2x
            self.tensorLR[index, 0] = LR

        
        self.set_attrs(total_len=len(self.image_filenames), batch_size=batch_size, shuffle=False)

    def __getitem__(self, index):
        return jt.Var(self.tensorLR[index]).float(), jt.Var(self.tensor2x[index]).float(), jt.Var(self.tensor4x[index]).float()
            
