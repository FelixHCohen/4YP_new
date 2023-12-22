import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from tqdm import tqdm
import time
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from data_aug.data import train_test_split
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss
from utils import *
import cv2
from wandb_train import get_data
device = "cuda:0"

def convert_to_bmp(mask,mask_path):
    mask = mask.numpy()
    #print(mask)
    #print(f'1 values: {np.where(mask==2)}')
    #print(np.where(mask==3))
    #
    mask = np.where(mask == 1, 128, mask)  # cup and disc pixels sset to 1
    mask = np.where(mask ==0,255, mask)  # background pixels set to 0
    mask = np.where(mask ==2,0, mask)  # set cup values to 2
    mask = mask.transpose((1,2,0))
    mask = np.repeat(mask,3,axis=2)
    cv2.imwrite(mask_path,mask)

def crop(output,np_image,label,i,dataset):
    img_path = f'/home/kebl6872/Desktop/new_data/REFUGE2/cropped/{dataset}/image'
    msk_path = f'/home/kebl6872/Desktop/new_data/REFUGE2/cropped/{dataset}/mask'
    create_dir(img_path)
    create_dir(msk_path)
    output = output.cpu()
    i_l = 0

    while i_l < output.size(2):
        if torch.any(torch.eq(output[:,:, i_l,:], 1)):
            left_column = i_l
            break
        i_l+=1

    i_r = output.size(2)-1

    while i_r > 0:
        if torch.any(torch.eq(output[:,:, i_r,:], 1)):
            right_column = i_r
            break
        i_r-=1

    j_t = 0

    while j_t < output.size(3):
        if torch.any(torch.eq(output[:,:,:,j_t], 1)):
            top_row = j_t
            break
        j_t += 1
    j_b = output.size(3) - 1

    while j_b > 0:
        if torch.any(torch.eq(output[:,:,:,j_b], 1)):
            bottom_row = j_b
            break
        j_b -= 1
    try:
        col_pad_width = 256 - (right_column-left_column)
    except:
        print("column error")
        return
    try:
        row_pad_width = 256 - (bottom_row - top_row)
    except:
        print("row error")
        return

    if col_pad_width < 0:
        left_col_final = left_column - col_pad_width//2
        right_col_final = right_column + (col_pad_width-col_pad_width//2)
    else:
        left_col_final = max(0,left_column - col_pad_width//2)
        new_col_pad_width = 256-(right_column-left_col_final)
        right_col_final = min(output.size(2),right_column + new_col_pad_width)
        if right_col_final == output.size(2):
            left_col_final = output.size(2)-256 #for case when right col is at edge

    if row_pad_width < 0:
        top_row_final = top_row - row_pad_width//2
        bottom_row_final = bottom_row + (row_pad_width-row_pad_width//2)
    else:

        top_row_final = max(0,top_row-row_pad_width//2) # deals with top row edge case
        new_row_pad_width = 256 - (bottom_row-top_row_final)
        bottom_row_final = min(output.size(3),bottom_row+new_row_pad_width)
        if bottom_row_final == output.size(3):
            top_row_final = output.size(3)-256 #for case when bottom row is at edge
                                           # both edge cases cannot occur simultaneously


    cv2.imwrite(f'{img_path}/image{i}.jpg',np_image[0,left_col_final:right_col_final,top_row_final:bottom_row_final,:])
    mask_path =  f'{msk_path}/mask{i}.bmp'
    #print(f'label shape: {label.shape}')
    convert_to_bmp(label[0, :, left_col_final:right_col_final, top_row_final:bottom_row_final],mask_path),



def crop_output(model,loader,dataset):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i,(image,label,np_image) in enumerate(loader):
            np_image=np_image.numpy()
            image_input = image.to(device,dtype=torch.float32)
            output = model(image_input).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            crop(output,np_image,label,i,dataset)


def make_cups(config,path):
    model = UNet(3, config["classes"], config["base_c"], config["kernels"], config["norm_name"])
    if not os.path.exists(path):
        print("path does not exist")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    model.eval()
    train,test = get_data(train=True,disc_only=False,crop=True),get_data(train=False,disc_only=False,crop=True)
    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=False, num_workers=4)

    return model,train_loader,test_loader
def crop_pipeline(config,model_path):
    model,train_loader,test_loader = make_cups(config,model_path)
    crop_output(model,train_loader,'train')
    crop_output(model,test_loader,'val')

seeding(42)
config = dict(classes=2,base_c=12,kernels=[6,12,24,48],norm_name='batch',batch_size=15,lr=5e-5,seed=37)
print(config)
seeding(config["seed"])
model_path = f'/home/kebl6872/Desktop/new_data/REFUGE2/test/disc_only/1600_unet_{config["norm_name"]}_lr_{config["lr"]}_bs_{config["batch_size"]}_fs_{config["base_c"]}_[{"_".join(str(k) for k in config["kernels"])}]/Checkpoint/seed/{config["seed"]}/lr_{config["lr"]}_bs_{config["batch_size"]}_lowloss.pth'
crop_pipeline(config,model_path)