from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import re

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_img(data):
    plt.figure(figsize=(10,10))
    img_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray)
    plt.show()

def resize_data(images,masks,save_path,name_add = '',pad=False):
    size = (512,512)
    size_reg = set()
    for idx, (x,y) in tqdm(enumerate(zip(images,masks))):

        name = x.split("/")[-1].split(".")[0]
        name = f'{name_add}{name}'

        img = cv2.imread(x,cv2.IMREAD_COLOR)
        mask = cv2.imread(y)

        delta = img.shape[1]-img.shape[0]
        size_reg.add(img.shape)
        d1 = delta//2
        d2 = delta - delta//2

        if not pad:
            img = img[:,d1:-1*d2,:] # cut off sides to make aspect ratio more similar to 1:1 to avoid distortion
            mask = mask[:,d1:-1*d2,:]
        else:
            img = np.pad(img, ((d1, d2), (0, 0), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((d1, d2), (0, 0), (0, 0)), mode='constant', constant_values=0)

        tmp_image_name = f"{name}.png"
        tmp_mask_name = f"{name}_mask.png"
        image_path = os.path.join(save_path, "image", tmp_image_name)
        mask_path = os.path.join(save_path, "mask", tmp_mask_name)
        img = cv2.resize(img,size)
        mask = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_path,mask)
        cv2.imwrite(image_path,img)

    print(size_reg)



create_dir('/Users/felixcohen/new_data/MESSIDOR/image/')
create_dir('/Users/felixcohen/new_data/MESSIDOR/mask/')
save_path = '/Users/felixcohen/new_data/MESSIDOR'
#

messidor_path = '/Users/felixcohen/Downloads/RIGA_images/MESSIDOR'
messidor_images  = sorted(glob(f'{messidor_path}/*prime.tif'))
messidor_masks_path = '/Users/felixcohen/Downloads/RIGA_masks/DiscCups/MESSIDOR/hards'
messidor_masks = sorted(glob(f'{messidor_masks_path}/*'))

br1_images = sorted(glob('/Users/felixcohen/Downloads/RIGA_images/BinRushed/BinRushed1-Corrected/*prime.jpg'))
print(br1_images)
br2_images = sorted(glob('/Users/felixcohen/Downloads/RIGA_images/BinRushed/BinRushed2/*prime.jpg'))
br3_images = sorted(glob('/Users/felixcohen/Downloads/RIGA_images/BinRushed/BinRushed3/*prime.jpg'))
br4_images = sorted(glob('/Users/felixcohen/Downloads/RIGA_images/BinRushed/BinRushed4/*prime.jpg'))
print(br4_images)
br1_masks = sorted(glob('/Users/felixcohen/Downloads/RIGA_masks/DiscCups/BinRushed/BinRushed1-Corrected/hards/*'))
br2_masks = sorted(glob('/Users/felixcohen/Downloads/RIGA_masks/DiscCups/BinRushed/BinRushed2/hards/*'))
br3_masks = sorted(glob('/Users/felixcohen/Downloads/RIGA_masks/DiscCups/BinRushed/BinRushed3/hards/*'))
br4_masks = sorted(glob('/Users/felixcohen/Downloads/RIGA_masks/DiscCups/BinRushed/BinRushed4/hards/*'))
br_images = [br1_images,br2_images,br3_images,br4_images]
br_masks = [br1_masks,br2_masks,br3_masks,br4_masks]
br_names = ['br1_','br2_','br3_','br4_']
save_paths = ['/Users/felixcohen/new_data/br1','/Users/felixcohen/new_data/br2','/Users/felixcohen/new_data/br3','/Users/felixcohen/new_data/br4']


magrabia_m_images = sorted(glob('/Users/felixcohen/Downloads/RIGA_images/Magrabia/MagrabiaMale/*prime.tif'))
magrabia_f_images = sorted(glob('/Users/felixcohen/Downloads/RIGA_images/Magrabia/MagrabiFemale/*prime.tif'))
magrabia_m_masks = sorted(glob('/Users/felixcohen/Downloads/RIGA_masks/DiscCups/Magrabia/MagrabiaMale/hards/*'))
magrabia_f_masks = sorted(glob('/Users/felixcohen/Downloads/RIGA_masks/DiscCups/Magrabia/MagrabiFemale/hards/*'))
magrabia_images = [magrabia_m_images,magrabia_f_images]
magrabia_masks = [magrabia_m_masks ,magrabia_f_masks ]
mag_names = ['m_','f_']
mag_paths = ['/Users/felixcohen/new_data/magrabia_m','/Users/felixcohen/new_data/magrabia_f']

for p in mag_paths:
    create_dir(f'{p}/image/')
    create_dir(f'{p}/mask/')
for imgs,masks,name_add,path in zip(magrabia_images,magrabia_masks,mag_names,mag_paths):
    resize_data(imgs,masks,path,name_add,pad=True)
# for sp in save_paths:
#     create_dir(f'{sp}/image/')
#     create_dir(f'{sp}/mask/')
# for imgs,masks,name_add,path in zip(br_images,br_masks,br_names,save_paths):
#     resize_data(imgs,masks,path,name_add)
#
#
#
#
#
# resize_data(messidor_images,messidor_masks,save_path)











