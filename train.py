import jittor as jt
from jittor import Module
from jittor import optim
from jittor.dataset import Dataset
from jittor import lr_scheduler

from lapsrn import *
from dataset import LapSRNDataset
from util import *
from test import test

import os
import argparse


parser = argparse.ArgumentParser(description='LapSRN Jittor Training')
parser.add_argument("--model", type=str, default="", help="Select a trained model")
parser.add_argument("--test_dataset", type=str, default="", help="Select a test dataset")
parser.add_argument("--dataset", type=str, default="", help="Select a dataset")
parser.add_argument("--batch", type=int, default=20, help="Enter a batch number (must be divisible by the number of data)")
parser.add_argument("--epochs", type=int, default=5, help="Every n epochs save the file")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--rclen", type=int, default=128,help="Random cut length, must be 2^n")
parser.add_argument("--step", type=float, default=1e-4,help="LR decay step")
parser.add_argument("--cuda", type=bool, default=False,help="use CUDA")
opt = parser.parse_args()



def prepare():

    global saved_models_dir, model, dataset, optimizer, loss, scheduler

    if opt.cuda:
        print("Using CUDA")
        jt.flags.use_cuda = 1

    saved_models_dir = "trained_models/"
    is_folder_exists = os.path.exists(saved_models_dir)
    if not is_folder_exists:
        os.makedirs(saved_models_dir) # Create a new directory because it does not exist 
        print("The saved models directory " + saved_models_dir + " is created!")

    model = Net()

    if(len(opt.model) > 0):
        model.load(opt.model)
    
    dataset = LapSRNDataset(opt.dataset, rc_len=opt.rclen, batch_size=20)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = LRScheduler(optimizer, opt.step)  # updates the learning rate over epochs
    loss = CharbonnierLoss()

def train(epoch):
    
    model.train()
    for LR, HR_2_target, HR_4_target in dataset:

        HR_2x, HR_4x = model(LR)

        loss_x2 = loss(HR_2x, HR_2_target)
        loss_x4 = loss(HR_4x, HR_4_target)
        
        optimizer.backward(loss=loss_x2)
        optimizer.step(loss_x4)

    print("=== Epoch:{}, Learning rate:{}".format(epoch, optimizer.lr))
        

def save_params(model, epoch):
    path = saved_models_dir + "lapsrn_model_epoch_{}.pkl".format(epoch)
    model.save(path)
    print("Model saved as: {}".format(path))
    return path


def main():

    

    prepare()
    best_psnr_diff = 0
    epoch = 0
    while(True):            
        epoch += 1

        # update learning rate  
        scheduler.step(epoch) 

        # train model with dataset
        train(epoch) 

        if epoch % opt.epochs == 0:
            save_path = save_params(model, epoch) # save model on disk as pkl
            pred_psnr, bc_psnr = test(save_path, opt.test_dataset)
            print(f"=== Prediction_PSNR = {pred_psnr}, Bicubic_PSNR = {bc_psnr}")

        # print(f'=== Epoch:{epoch} PSNR={psnr_predicted:.4f}, Best={best_psnr:.4f}')

if __name__ == "__main__":
    main()