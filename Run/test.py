import torch
import torch.cuda as gc
import cv2
from train_neat import *
from PromptUNet.PromptUNet import PromptUNet,pointLoss,NormalisedFocalLoss,combine_loss,combine_point_loss
from utils import *
import glob
from monai.losses import DiceLoss

print("Installed Torch's Cuda version: ", torch.version.cuda)

available = gc.is_available()
numGPU = gc.device_count()

print("GPU Availability: ", available, "Number of GPUs: ", numGPU, sep='\n')
for i in range(numGPU):
    print("Device Name:", gc.get_device_name(i), "Device Capability:", gc.get_device_capability(i), sep='\t')