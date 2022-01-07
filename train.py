import jittor as jt
import os
from jittor import Module

from jittor import optim
from jittor.dataset import Dataset
from jittor import lr_scheduler

from lapsrn import *
from dataset import LapSRNDataset
from util import *

def prepare():

    global saved_models_dir, model, dataset, optimizer, loss, scheduler

    saved_models_dir = "trained_models/"
    is_folder_exists = os.path.exists(saved_models_dir)
    if not is_folder_exists:
        os.makedirs(saved_models_dir) # Create a new directory because it does not exist 
        print("The saved models directory " + saved_models_dir + "is created!")

    model = Net()
    
    dataset = LapSRNDataset("SR_training_datasets/General100", batch_size=20)
    optimizer = optim.Adam(model.parameters(), lr=1/10000)
    scheduler = LRScheduler(optimizer, 1/10000)  # updates the learning rate over epochs
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


def main():
    prepare()
    best_psnr = 0
    epoch = 0
    while(True):            
        epoch += 1

        # update learning rate  
        scheduler.step(epoch) 

        # train model with dataset
        train(epoch) 

         # evaluate


        if epoch % 10 == 0:
            save_params(model, epoch) # save model on disk as pkl

        # print(f'=== Epoch:{epoch} PSNR={psnr_predicted:.4f}, Best={best_psnr:.4f}')

if __name__ == "__main__":
    main()