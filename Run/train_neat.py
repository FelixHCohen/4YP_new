import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import time
from torch.utils.data import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from data_aug.data import train_test_split,GS1_dataset
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss, DiceFocalLoss
from utils import *
from random import randint


def train_batch(images,labels,model,optimizer,criterion):
    images,labels = images.to(device,dtype=torch.float32),labels.to(device,dtype=torch.float32)
    outputs = model(images)
    loss = criterion(outputs,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
def train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct + 1).zfill(5)} batches: {loss:.3f}")

def save_model(path,name):
    artifact = wandb.Artifact(name=name, type="model")
    # Add the model file to the artifact
    artifact.add_file(path)
    # Log the artifact as an output of the run
    wandb.run.log_artifact(artifact)

def test(model,test_loader,criterion,config,best_valid_score,example_ct):
    model.eval()

    with torch.no_grad():
        val_score = 0
        f1_score_record = np.zeros(4)
        total = 0
        for images,labels in test_loader:
            images,labels = images.to(device,dtype=torch.float32),labels.to(device)
            outputs = model(images).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            score = criterion(outputs,labels)
            val_score += score[1].item()/2 + score[2].item()/2
            f1_score_record += score
            total += labels.size(0)
        f1_score_record /= len(test_loader) #currently only works with batchsize=1
        val_score /= len(test_loader)
        print(f"model tested on {total} images" +
                  f"val_score: {val_score} f1_scores {f1_score_record}")
        wandb.log({"val_score": val_score, "Validation Background F1":f1_score_record[0],"Validation Outer Ring F1":f1_score_record[1],
                   "Validation Cup F1": f1_score_record[2],"Validation Disk F1": f1_score_record[3]},step=example_ct)


        #     # Save the model in the exchangeable ONNX format
        # torch.onnx.export(model, images, "model.onnx",opset_version=16)
        # wandb.save("model.onnx")

    model.train()

    # if val_score > best_valid_score[0]:
    #     data_str = f"Valid score improved from {best_valid_score[0]:2.8f} to {val_score:2.8f}. Saving checkpoint: {checkpoint_path_lowloss}"
    #     print(data_str)
    #     best_valid_score[0] = val_score
    #     torch.save(model.state_dict(), config.low_loss_path)
    #     save_model(config.low_loss_path,"low_loss_model")

    return f1_score_record[2],f1_score_record[3]



def train(model, loader,test_loader, criterion, eval_criterion, config):

    wandb.watch(model,criterion,log='all',log_freq=50) #this is freq of gradient recordings

    example_ct = 0
    batch_ct = 0
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_valid_score = [0.0]#in list so I can alter it in test function
    for epoch in tqdm(range(config.epochs)):

        avg_epoch_loss = 0.0
        start_time = time.time()
        for _,(images, labels) in enumerate(loader):

            loss = train_batch(images,labels,model,optimizer,criterion)
            avg_epoch_loss += loss
            example_ct += len(images)
            batch_ct +=1

            if ((batch_ct+1)%4)==0:
                train_log(loss,batch_ct,epoch)



        cup_loss,disk_loss = test(model,test_loader,eval_criterion,config,best_valid_score,batch_ct)
        avg_epoch_loss/=len(loader)
        end_time = time.time()
        iteration_mins,iteration_secs = train_time(start_time,end_time)
        data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
        data_str += f'\t Val Cup: {cup_loss:.8f}\n'
        data_str += f'\t Val Disk: {disk_loss:.8f}\n'
        print(data_str)
    torch.save(model.state_dict(),config.low_loss_path)
    #save_model(config.final_path,"final_model")




def get_data(train,gs1=False,rim=False,refuge_test=False,transform=False,return_path=False,gamma=False):
    if gs1:
        return get_gs1_or_rim_data(train,transform)

    if rim:
        return get_gs1_or_rim_data(train,transform,rim=True)

    if refuge_test:
        #/Users/felixcohen /home/kebl6872/
        x = sorted(glob(f"/users/kebl6872/Desktop/new_data/REFUGE2/test/image/*"))
        y = sorted(glob(f"/users/kebl6872/Desktop/new_data/REFUGE2/test/mask/*"))
        print(f'testing dataset of size: {len(x)}')
    else:
        if train:
            dataset_type = 'train'
        if train == False and gamma==False:
            dataset_type= 'val'
        if train == False and gamma ==True:
            dataset_type='test'

        if gamma == True:
            dataset = 'Gamma'
        else:
            dataset = 'REFUGE2'

        x = sorted(glob(f"/users/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/image/*"))
        y = sorted(glob(f"/users/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/mask/*"))
        data_str = f"Training dataset size: {len(x)}"
        print(data_str)



    dataset = train_test_split(x,y,transform=transform,return_path=return_path)


    return dataset

# def get_riga_data(transform,messidor=True,br=False,mag=False)
#     if messidor:
#         dataset = 'MESSIDOR'
#     elif br:

def get_gs1_or_rim_data(train,transform,rim=False,):
    if train:
        dataset_type = 'train'
    else:
        dataset_type = 'test'

    if rim:
        dataset = "RIMDL"
    else:
        dataset = "GS1"
#/home/kebl6872/Desktop/
    gs1_x = sorted(glob(f"/users/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/image/*"))
    gs1_c = sorted(glob(f"/users/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/cup_mask/*"))
    gs1_d = sorted(glob(f"/users/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/disc_mask/*"))
    data_str = f"{dataset_type} dataset size: {len(gs1_x)}"
    print(data_str)

    dataset = GS1_dataset(gs1_x,gs1_c,gs1_d,transform=transform,disc_only=False)

    return dataset
def make_loader(dataset,batch_size):
    loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,)
    return loader
def make(config):
    if config.dataset=="GS1":
        gs1 = True
    else:
        gs1 = False

    if config.dataset == "GAMMA":
        gamma=True
    else:
        gamma=False
    train,test = get_data(train=True,transform=config.transform,gs1=gs1,gamma=gamma),get_data(train=False,gs1=gs1,gamma=gamma)
    eval_criterion = f1_valid_score
    train_loader = DataLoader(dataset=train,batch_size=config.batch_size,shuffle=True,)
    test_loader = DataLoader(dataset=test,batch_size=1,shuffle=False)
    criterion =  DiceFocalLoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=1.8,lambda_focal=1.0,gamma=5)
    criterion = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5,lambda_ce=0.5)

    model = UNet(3,config.classes,config.base_c,config.kernels,config.norm_name)


    return model,train_loader,test_loader,criterion,eval_criterion
def model_pipeline(hyperparameters):
    with wandb.init(project="Weak_UNet_experiment",config=hyperparameters, dir = '/data/engs-mlmi1/kebl6872/wandb'):
        config = wandb.config

        model,train_loader,test_loader,criterion,eval_criterion = make(config)
        # print(model)
        model = model.to(device)
        train(model,train_loader,test_loader,criterion,eval_criterion,config)

    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Specify Parameters')

    parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
    parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
    parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
    parser.add_argument('--base_c', metavar='--base_c', default=12, type=int,
                        help='base_channel which is the first output channel from first conv block')

    parser.add_argument('no_runs', metavar='no_runs', type=int,
                        help='how many random seeds you want to run experiment on')

    args = parser.parse_args()
    lr, batch_size, gpu_index, no_runs = args.lr, args.b_s, args.gpu_index, args.no_runs
    base_c = args.base_c
    norm_name = 'batch'
    model_name = 'unet'
    print(torch.cuda.is_available())
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')

    for a in ['REFUGE']:
        for _ in range(no_runs):
            config = dict(epochs=60, classes=3, base_c = 12, kernels=[6,12,24,48], norm_name=norm_name,
                          batch_size=batch_size, learning_rate=lr, dataset=a,
                          architecture=model_name,seed=401,transform=True)
            config["seed"] = randint(801,1000)
            seeding(config["seed"])


            data_save_path = f'/home/kebl6872/Desktop/new_data/{config["dataset"]}/test/{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{config["base_c"]}_[{"_".join(str(k) for k in config["kernels"])}]/'


            #haHAVE ADDED THIS TO TRAIN 'WEAK' UNET FOR PROMPT TRAINING
            data_save_path = f'/home/kebl6872/Desktop/weakunet/'

            create_dir(data_save_path + f'Checkpoint/seed/{config["seed"]}')
            checkpoint_path_lowloss = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_lowloss.pth'
            checkpoint_path_final = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_final.pth'
            create_file(checkpoint_path_lowloss)
            #create_file(checkpoint_path_final)
            config['low_loss_path']=checkpoint_path_lowloss
            #config['final_path'] = checkpoint_path_final

            model = model_pipeline(config)




# nohup python3 train.py.... optioons.... &>> out_file.txt &
#or
# nohup python3 train.py.... optioons.... >>& out_file.txt &