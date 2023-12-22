from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

im_path = '/data_hd1/data/eyes/RIM-ONE_DL/RIM-ONE_DL_images/partitioned_by_hospital'
mask_path =  '/data_hd1/data/eyes/RIM-ONE_DL/RIM-ONE_DL_reference_segmentations'
test_im_path = f'{im_path}/test_set'
train_im_path = f'{im_path}/training_set'

train_x = glob(f'{train_im_path}/**/*.png',recursive=True)
test_x = glob(f'{test_im_path}/**/*.png',recursive=True)
all_y = sorted(glob(f'{mask_path}/**/*.png',recursive=True),key = lambda x: x.split("/")[-1].split(".")[0])
train_register = set()
test_register = set()

for x in train_x:
    train_register.add(x.split("/")[-1].split(".")[0])

for x in test_x:
    test_register.add(x.split("/")[-1].split(".")[0])

all_x = sorted(train_x + test_x, key = lambda x: x.split("/")[-1].split(".")[0])


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def resize_data(images, masks, save_path):
    size = (512, 512)
    disc_masks = masks[1::2]
    cup_masks = masks[::2]

    paired_masks = [[a,b] for a,b in zip(disc_masks,cup_masks)]
    for image_path,(disc_path,cup_path) in zip(images,paired_masks):
        name = image_path.split("/")[-1].split(".")[0]

        if name in test_register:
            t = 'test'
        elif name in train_register:
            t = 'train'
        else:
            print(f'{name} not in registers')
            continue

        img = cv2.imread(image_path)
        img = cv2.resize(img,size)

        y_disc = cv2.imread(disc_path)
        y_disc = cv2.resize(y_disc,size)


        y_cup = cv2.imread(cup_path)
        y_cup = cv2.resize(y_cup,size)


        cv2.imwrite(f'{save_path}/{t}/image/{name}.png',img)
        cv2.imwrite(f'{save_path}/{t}/disc_mask/{name}.png',y_disc)
        cv2.imwrite(f'{save_path}/{t}/cup_mask/{name}.png',y_cup)



create_dir('/home/kebl6872/Desktop/new_data/RIMDL/train/image/')
create_dir('/home/kebl6872/Desktop/new_data/RIMDL/train/cup_mask/')
create_dir('/home/kebl6872/Desktop/new_data/RIMDL/train/disc_mask/')
create_dir('/home/kebl6872/Desktop/new_data/RIMDL/test/image/')
create_dir('/home/kebl6872/Desktop/new_data/RIMDL/test/cup_mask/')
create_dir('/home/kebl6872/Desktop/new_data/RIMDL/test/disc_mask/')
save_path = '/home/kebl6872/Desktop/new_data/RIMDL'

#resize_data(all_x,all_y,save_path)


