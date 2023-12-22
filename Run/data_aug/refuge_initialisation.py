import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations import HorizontalFlip, VerticalFlip, Rotate


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path_train_x, path_train_y, path_val_x, path_val_y, path_test_x, path_test_y):
    train_x = sorted(glob(os.path.join(path_train_x,'**', '*.jpg'),recursive=True)) #glob returns array of all paths matching path*.jpg - sorted will line up masks and images as they will all only differ by their name
    train_y = sorted(glob(os.path.join(path_train_y,'**', '*.bmp'),recursive=True))
    val_x   = sorted(glob(os.path.join(path_val_x, '*.jpg')))
    val_y   = sorted(glob(os.path.join(path_val_y, '*.bmp')))
    test_x  = sorted(glob(os.path.join(path_test_x, '*.jpg')),key = lambda x: x.split('/')[-1])
    test_y  = sorted(glob(os.path.join(path_test_y,'**', '*.bmp'),recursive=True),key = lambda x: x.split('/')[-1])

    print(glob(os.path.join(path_train_x, '*.jpg')))
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def augment_data(images, masks, save_path, augment=True):
    size = (256, 256)
    img_reg = set()
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0] # extracts name from path/name.jpg

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y)

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=[0,23], p=1.0,border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=(255,255,255))
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug = Rotate(limit=[-23,0], p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=(255, 255, 255))
            augmented = aug(image=x, mask=y)


            x4 = augmented["image"]

            y4 = augmented["mask"]
            X = [x, x1, x2, x3,x4]
            Y = [y, y1, y2, y3,y4]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):


            tmp_image_name = f"{name}_{index}.jpg"
            tmp_mask_name = f"{name}_{index}.bmp"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)
            i = cv2.resize(i, size)
            m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    ori_data_path = '/home/kebl6872/Desktop/refuge-challenge/REFUGE2-Training/REFUGE2-Training/'
    data_path_train_x = ori_data_path + "REFUGE1-Train-400/Images"
    data_path_train_y = ori_data_path + "REFUGE1-Train-400/Annotation-Training400/Disc_Cup_Masks"
    data_path_val_x   = ori_data_path + "REFUGE1-Val-400/REFUGE-Validation400"
    data_path_val_y   = ori_data_path + "REFUGE1-Val-400/REFUGE-Validation400-GT/Disc_Cup_Masks"
    data_path_test_x  = ori_data_path + "REFUGE1-Test-400/Images"
    data_path_test_y  = ori_data_path + "REFUGE1-Test-400/Annotation/Disc_Cup_Masks"
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(data_path_train_x, data_path_train_y,
                                                                      data_path_val_x, data_path_val_y,
                                                                      data_path_test_x, data_path_test_y)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Val: {len(val_x)} - {len(val_y)}")
    print(f"Train: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("/home/kebl6872/Desktop/new_data/REFUGE256/train/image/")
    create_dir("/home/kebl6872/Desktop/new_data/REFUGE256/train/mask/")
    create_dir("/home/kebl6872/Desktop/new_data/REFUGE256/val/image/")
    create_dir("/home/kebl6872/Desktop/new_data/REFUGE256/val/mask/")
    create_dir("/home/kebl6872/Desktop/new_data/REFUGE256/test/image/")
    create_dir("/home/kebl6872/Desktop/new_data/REFUGE256/test/mask/")

    """ Data augmentation"""
    # set to True to increase training dataset, thus to recude overfitting

    augment_data(train_x, train_y, "/home/kebl6872/Desktop/new_data/REFUGE256/train/", augment=False)
    augment_data(val_x, val_y, "/home/kebl6872/Desktop/new_data/REFUGE256/val/", augment=False)
    augment_data(test_x, test_y, "/home/kebl6872/Desktop/new_data/REFUGE256/test/", augment=False)
