from cv2 import imread
from lapsrn import Net
from util import *
from upscale import upscale

import os
import argparse

def test(model, test_folder):

    # metrics
    avg_psnr_pred = 0.0
    avg_psnr_bicubic = 0.0
    avg__time = 0.0

    # get test points, x4 resized data with bicubic 
    image_filenames = [os.path.join(test_folder, x) for x in os.listdir(test_folder) if is_image_file(x)]

    for image_path in image_filenames:

        # get compressed version 
        GT_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
        LR_img = scale_lr(GT_img).astype(float)
        BC_img = cv2.resize(LR_img, (GT_img.shape[1], GT_img.shape[0]), interpolation=cv2.INTER_CUBIC).astype(float)

        avg_psnr_bicubic += compute_psnr(BC_img, GT_img, 4)

        # run the model
        res_img = upscale(LR_img, 4, model)

        avg_psnr_pred += compute_psnr(res_img, GT_img, 4)

    print("Dataset=", test_folder.split('/')[-1])
    print("PSNR_predicted=", avg_psnr_pred/len(image_filenames))
    print("PSNR_bicubic=", avg_psnr_bicubic/len(image_filenames))

    return avg_psnr_pred/len(image_filenames), avg_psnr_bicubic/len(image_filenames)


if __name__ == "__main__":

    # parse params
    parser = argparse.ArgumentParser(description='LapSRN Jittor Evaluation')
    parser.add_argument("--model", type=str, default="./model_best.pkl", help="Select a trained model")
    parser.add_argument("--dataset", type=str, default="", help="Select a dataset")
    parser.add_argument("--cuda", type=bool, default=False,help="use CUDA")

    opt = parser.parse_args()
    print(opt)

    # some arg checks
    if len(opt.dataset) == 0:
         raise Exception("Please select a dataset with --dataset")
        
    if opt.cuda:
        print("Using CUDA")
        jt.flags.use_cuda = 1

    test(opt.model, opt.dataset)