import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import (
    Compose,
    CLAHE,
    ColorJitter,
    Equalize,
    FancyPCA,
    HueSaturationValue,
    ISONoise,
    ImageCompression,
    MultiplicativeNoise,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise,
    Blur,
    MedianBlur,
    RGBShift,
    ChannelShuffle,
)

def intensity_aug(p=0.8):
    return Compose([
        CLAHE(p=0.1),
        ColorJitter(p=0.2),
        Equalize(p=0.05),
        FancyPCA(p=0.1),
        HueSaturationValue(p=0.1),
        ISONoise(p=0.1),
        MultiplicativeNoise(p=0.1),
        RandomBrightnessContrast(p=0.3),
        RandomGamma(p=0.2),
        GaussNoise(p=0.1),
        Blur(p=0.1),
        MedianBlur(p=0.01),
        #RGBShift(p=0.1),
        #ChannelShuffle(p=0.05),
        A.HorizontalFlip(p=0.25),
        A.Rotate(limit=270,p=0.4,border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0)
    ], p=p)

basic_transform = A.Compose([A.VerticalFlip(p=0.1),A.HorizontalFlip(p=0.25),A.Rotate(limit=45, p=0.55,border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0)
           ]) #prev probs were 0.15 0.45, 25 degree limit, for gs1/gamma 0.25,0.6,25 degree limit
class train_test_split(Dataset):
    def __init__(self, images_path, masks_path, disc_only=False,transform=False,return_path=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.num_samples = len(images_path)
        self.disc_only = disc_only
        self.return_path = return_path
        if transform:
            self.transform = intensity_aug()
        else:
            self.transform = False


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask < 128, 2, mask)  # set cup values to 2
        mask = np.where(mask == 128, 1, mask)  # disc pixels sset to 1
        mask = np.where(mask > 128, 0, mask)  # background pixels set to 0

        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """


        if self.disc_only:
            mask = np.where(mask == 2, 1, mask)  # set cup values to 2

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)
        if self.return_path == True:
            res = image, mask,self.images_path[index],self.masks_path[index]
        else:
            res = image,mask

        return res

    def __len__(self):
        return self.num_samples

class crop_dataset(train_test_split):

    def __init__(self,images_path,masks_path,disc_only=False,transform=False):
        super().__init__(images_path,masks_path,disc_only)

    def get_np_image(self,index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)

        return image

    def __getitem__(self,index):
        np_image = self.get_np_image(index)
        image,mask = super().__getitem__(index)

        return image,mask,np_image

class cup_dataset(train_test_split):
    def __init__(self, images_path, masks_path, disc_only=False,transform=False):
        super().__init__(images_path, masks_path, disc_only)


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        '''Normalise tensity in range [-1,-1]'''
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask < 128, 1, mask)  # set cup values to 1
        mask = np.where(mask >= 128, 0, mask)  # background pixels set to 0



        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)  # (1,512,512)

        return image, mask


class GS1_dataset(Dataset):
    def __init__(self, images_path, cup_path,disc_path, disc_only=False,transform=False):

        self.images_path = images_path
        self.cup_path = cup_path
        self.disc_path = disc_path
        self.num_samples = len(images_path)
        self.disc_only = disc_only
        if transform:
            self.transform = basic_transform
        else:
            self.transform = False


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        disc_mask = cv2.imread(self.disc_path[index], cv2.IMREAD_GRAYSCALE)
        cup_mask = cv2.imread(self.cup_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(disc_mask >= 191, 1, disc_mask)  # set disc values to 1 w 75% agreement
        mask = np.where(disc_mask < 191, 0, mask)  # background to 0

        mask = np.where(cup_mask >= 191, 2, mask)  # set cup values to 2
        if self.disc_only:
            mask = np.where(mask>=2,1,mask)
        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """




        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)

        return image, mask

    def __len__(self):
        return self.num_samples


class RIMDL_dataset(GS1_dataset):

    def __init__(self,images_path,disc_path,cup_path,disc_only=False,transform=False):
        super().__init__(images_path,disc_path,cup_path,disc_only,transform)

    def __getitem__(self,index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        disc_mask = cv2.imread(self.disc_path[index], cv2.IMREAD_GRAYSCALE)
        cup_mask = cv2.imread(self.cup_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(disc_mask >= 255, 1, disc_mask)  # set disc vals to 1
        mask = np.where(disc_mask < 255, 0, mask)  # background to 0

        if not self.disc_only:
            mask = np.where(cup_mask >= 255, 2, mask)  # set cup values to 2
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)

        """ Reading mask """

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)  # (1,512,512)

        return image, mask



