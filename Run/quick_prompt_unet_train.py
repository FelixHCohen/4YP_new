from data_aug.data import small_prompt_dataset
from torch.utils.data import DataLoader
from UNET.PromptUNet import PromptUNet
from monai.losses import DiceCELoss
import pandas as pd
from utils import *

dataset = small_prompt_dataset(pd.read_csv('/home/kebl6872/Desktop/new_data/small_prompt_set/df.csv'))
dataloader = DataLoader(dataset,batch_size=10)
criterion = DiceCELoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PromptUNet(device,3, 1, kernels=[2, 4, 6, 8], base_c=12,).to(device)
optimiser = torch.optim.Adam(model.parameters(),5e-5)
for epoch in range(100):
    for (images,masks,points) in dataloader:
        prompt_input = torch.tensor([[(1, 2, 1), (0, 0, 1)], [(1, 2, 1), (0, 0, 1)]])
        print(prompt_input.shape)
        images = images.to(device,dtype=torch.float32)
        print(f'im: {images.shape}')
        print(f'm: {masks.shape}')
        print(f' p: {points.shape}')
        points = points.to(device,dtype=torch.int)
        masks = masks.to(device)
        outputs = model(images,points)
        loss = criterion(outputs,masks)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(loss)

df = pd.read_csv('/home/kebl6872/Desktop/new_data/small_prompt_set/df.csv')
print(df["mask_path"][0][2:-3])