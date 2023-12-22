import numpy as np
import cv2
import os
from glob import glob
from GS1_initialiisaton import create_dir
from tqdm import tqdm

path = '/data_hd1/data/eyes/gamma_2021/task3_disc_cup_segmentation/training'
image_path = os.path.join(path,'fundus color images')
mask_path = os.path.join(path,'Disc_Cup_Mask')
train_x = sorted(glob(os.path.join(image_path,'*')))
train_y = sorted(glob(os.path.join(mask_path,'*')))

def resize_data(images,masks,save_path):
    size = (256,256)
    size_reg = set()
    for idx, (x,y) in tqdm(enumerate(zip(images,masks))):
        name = x.split("/")[-1].split(".")[0]

        img = cv2.imread(x,cv2.IMREAD_COLOR)
        delta = img.shape[1]-img.shape[0]
        size_reg.add(img.shape)
        d1 = delta//2
        d2 = delta - delta//2
        img = img[:,d1:-1*d2,:] # cut off sides to make aspect ratio more similar to 1:1 to avoid distortion

        mask = cv2.imread(y)
        mask = mask[:,d1:-1*d2,:]
        tmp_image_name = f"{name}.png"
        tmp_mask_name = f"{name}_mask.png"
        image_path = os.path.join(save_path, "image", tmp_image_name)
        mask_path = os.path.join(save_path, "mask", tmp_mask_name)
        img = cv2.resize(img,size)
        mask = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_path,mask)
        cv2.imwrite(image_path,img)

    print(size_reg)



create_dir('/home/kebl6872/Desktop/new_data/Gamma256/train/image/')
create_dir('/home/kebl6872/Desktop/new_data/Gamma256/train/mask/')
save_path = '/home/kebl6872/Desktop/new_data/Gamma256/train'
#
resize_data(train_x,train_y,save_path)


