import torch
from torch.utils.data import DataLoader
import albumentations as A
from data import *
from glob import glob
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys


'''quick script to visualise dataloader outputs'''
def tensor_to_numpy(tensor,mask=False):
    tensor = tensor[0,:,:,:].permute(1, 2, 0)
    tensor = tensor.numpy()
    '''this makes the images more visible to huma eye currently commented out'''
    if not mask:
        tensor = ((tensor*127.5)-127.5).astype(np.uint8)
    else:
         mask =np.where(mask==1,128,mask)
         mask= np.where(mask==2,255,mask)
         mask = mask.astype(np.uint8)
         mask = np.repeat(mask,3,3)
    return tensor

x = sorted(glob(f"/home/kebl6872/Desktop/new_data/REFUGE2/test/image/*"))
y = sorted(glob(f"/home/kebl6872/Desktop/new_data/REFUGE2/test/mask/*"))
print(x)
print(y)
gs1_x = sorted(glob(f"/home/kebl6872/Desktop/new_data/GS1/test/image/*"))
gs1_c = sorted(glob(f"/home/kebl6872/Desktop/new_data/GS1/test/cup_mask/*"))
gs1_d = sorted(glob(f"/home/kebl6872/Desktop/new_data/GS1/test/disc_mask/*"))

rim_x = sorted(glob(f"/home/kebl6872/Desktop/new_data/RIMDL/test/image/*"))
rim_c = sorted(glob(f"/home/kebl6872/Desktop/new_data/RIMDL/test/cup_mask/*"))
rim_d = sorted(glob(f"/home/kebl6872/Desktop/new_data/RIMDL/test/disc_mask/*"))

num = 4
vis_dataset = train_test_split(x[:num],y[:num],transform=True)
vis_loader = DataLoader(dataset=vis_dataset,batch_size=1,shuffle=False)
gs1_dataset = GS1_dataset(gs1_x[:num],gs1_c[:num],gs1_d[:num],transform=True,disc_only=False)
gs1_loader = DataLoader(dataset=gs1_dataset,batch_size=1,shuffle=True)
rim_dataset = RIMDL_dataset(rim_x[:num],rim_c[:num],rim_d[:num],transform=True)
rim_loader = DataLoader(rim_dataset,batch_size=1,shuffle=True)
images = list()
masks = list()
for i, (image,mask) in enumerate(gs1_loader):
    images.append(tensor_to_numpy(image))
    masks.append(tensor_to_numpy(mask))
    print(np.unique(mask))
    print(np.argwhere(mask==2))


# Define the number of rows and columns for the subplot
rows = num
cols = 2

# Create a figure with the specified size
plt.figure(figsize=(15, 15))

# Loop through the list of images and masks
for i, (image,mask) in enumerate(zip(images,masks)):

  # Plot the image in the odd-numbered subplot
  plt.subplot(rows, cols, 2 * i + 1)
  plt.imshow(images[i])
  plt.title(f'Image {i + 1}')
  plt.axis('off')
  # Plot the mask in the even-numbered subplot
  plt.subplot(rows, cols, 2 * i + 2)
  plt.imshow(masks[i])
  plt.title(f'Mask {i + 1}')
  plt.axis('off')

# Show the plot
plt.show()




