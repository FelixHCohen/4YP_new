import cv2

from train_neat import *
from PromptUNet.PromptUNet import PromptUNet,pointLoss,NormalisedFocalLoss,combine_loss,combine_point_loss
from utils import *
from test_on_diff_data import plot_output
import glob
from monai.losses import DiceLoss

def prompt_train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"attention training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct + 1).zfill(5)} batches: {loss:.3f}")
def prompt_train_batch_iterative(images,labels,points,point_labels,prev_output,model,optimizer,criterion,plot=False):
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    point_input,point_label_input = torch.from_numpy(points).to(device,dtype=torch.float),torch.from_numpy(point_labels).to(device,dtype=torch.float)
    start = time.perf_counter()
    outputs = model(images,point_input,point_label_input,train_attention=True)
    end = time.perf_counter()
    print(f'inference time: {end-start}')
    loss = criterion(outputs, labels)
    if plot:
        point_tuples = [(i,j,val[0]) for (i,j),val in zip(points[0,:,:],point_labels[0,:,:])]
        print(f'point_tuples: {point_tuples}')
        plot_output_and_prev(prev_output.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1),outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1),images,labels,loss,point_tuples,detach=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    new_points,new_point_labels = generate_points_batch(labels,outputs,detach=True)

    point_labels = np.concatenate([point_labels,new_point_labels],axis=1)
    points = np.concatenate([points,new_points],axis=1)
    return loss,points,point_labels,outputs
def prompt_train_batch(images,labels,points,point_labels,weak_unet_preds,model,optimizer,criterion,config,plot=False):
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    point_input, point_label_input = torch.from_numpy(points).to(device, dtype=torch.float), torch.from_numpy(point_labels).to(device, dtype=torch.float)
    #start = time.perf_counter()
    outputs = model(images, point_input, point_label_input, train_attention=True)
    #end = time.perf_counter()
    #print(f'batch inference time: {end-start}')
    #start = time.perf_counter()


    loss = criterion(outputs, labels,point_input,point_label_input,config.device)


    #end = time.perf_counter()
    #print(f'loss calculation time: {end-start}')
    if plot:
        image_idx = random.randint(0,images.shape[0]-1)
        point_tuples = [(i, j, val[0]) for (i, j), val in zip(points[image_idx, :, :], point_labels[image_idx, :, :])]
        print(f'point_tuples: {point_tuples}')
        plot_output_and_prev(weak_unet_preds[image_idx],
                             outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1), images, labels, loss, point_tuples,
                             detach=False,image_idx=image_idx)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def initial_test(images,labels,model,criterion,):
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    outputs = model(images, [], [], train_attention=False)
    points, point_labels = generate_points_batch(labels, outputs)
    point_tuples = [(i, j, val[0]) for (i, j), val in zip(points[0, :, :], point_labels[0, :, :])]

    outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
    score = criterion(outputs,labels)
    plot_output(outputs,images,labels,score[1],point_tuples)
def prompt_test(model,test_loader,criterion,config,best_valid_score,example_ct,num_points=6,plot=False):
    model.eval()

    with torch.no_grad():

        val_scores = np.zeros(num_points)
        f1_score_record = np.zeros((4,num_points))
        total = 0
        for _,(images,labels) in enumerate(test_loader):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            prev_output = torch.from_numpy(np.zeros((config.batch_size,1,512,512))).to(config.device)# need to make first input appear to havec come from model so it works w gen_points func
            points, point_labels = generate_points_batch(labels,prev_output ,num=1)
            #prev_output = prev_output.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            # above line was for when prev_output came out of model
            for i in range(num_points):

                point_input, point_label_input = torch.from_numpy(points).to(device,dtype=torch.float32), torch.from_numpy(point_labels).to(device, dtype=torch.float32)
                start = time.perf_counter()
                outputs = model(images, point_input, point_label_input,train_attention=True)
                end = time.perf_counter()
                print(f'test model inference: {end-start}')
                new_points, new_point_labels = generate_points_batch(labels, outputs)
                p_end = time.perf_counter()
                print(f'point gen time: {p_end - end}')
                outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
                score = criterion(outputs,labels)


                val_score = score[1].item() / 2 + score[2].item() / 2


                val_scores[i] += val_score

                print(f'point {i+1}: {val_score}')
                f1_score_record[:,i] += score
                plot = False
                if plot:

                    image_idx = random.randint(0, images.shape[0] - 1)
                    point_tuples = [(i, j, val[0]) for (i, j), val in
                                    zip(points[image_idx, :, :], point_labels[image_idx, :, :])]
                    print(f'point_tuples: {point_tuples}')
                    plot_output_and_prev(prev_output[image_idx,:,:,:].unsqueeze(dim=0).detach(),
                                         outputs, images, labels, score,
                                         point_tuples,
                                         detach=False, image_idx=image_idx)
                prev_output = outputs
                point_labels = np.concatenate([point_labels, new_point_labels], axis=1)
                points = np.concatenate([points, new_points], axis=1)

            total += labels.size(0)

    f1_score_record /= total
    val_scores /= total
    val_score_str = ', '.join([format(score,'.8f') for score in val_scores])
    disc_scores = ', '.join([format(score,'.8f') for score in f1_score_record[3,:]])
    cup_scores = ', '.join([format(score,'.8f') for score in f1_score_record[2, :]])

    return_str = f"model tested on {total} images\nval_scores: {val_score_str}\ndisc f1 scores {disc_scores}\ncup scores: {cup_scores}"


    data_to_log = {}




    # Loop through the validation scores and add them to the dictionary
    for i, val_score in enumerate(val_scores):
        data_to_log[f"val_score {i + 1} points"] = val_score
        data_to_log[f"Validation Background F1 Score {i + 1}"] = f1_score_record[0][i]
        data_to_log[f"Validation Disc F1 Score {i + 1}"] = f1_score_record[3][i]
        data_to_log[f"Validation Cup F1 Score {i + 1}"] = f1_score_record[2][i]
        data_to_log[f"Validation Outer Ring F1 Score {i + 1}"] = f1_score_record[1][i]

    wandb.log(data_to_log,step=example_ct)
    model.train()

    if val_scores[-1] > best_valid_score[0]:
        data_str = f"Valid score for point {len(val_scores)} improved from {best_valid_score[0]:2.8f} to {val_scores[-1]:2.8f}. Saving checkpoint: {config.low_loss_path}"
        print(data_str)
        best_valid_score[0] = val_scores[-1]
        torch.save(model.state_dict(), config.low_loss_path)
        # save_model(config.low_loss_path, "low_loss_model")

    return return_str



def prompt_train(model, loader,test_loader, criterion, eval_criterion, config,num_points=5):
    unet_path = "/home/kebl6872/Desktop/new_data/REFUGE/test/unet_batch_lr_0.0003_bs_16_fs_12_[6_12_24_48]/Checkpoint/seed/279/lr_0.0003_bs_16_lowloss.pth"
    #unet_path = "/home/kebl6872/Desktop/new_data/GAMMA/test/promptunet_batch_lr_0.0005_bs_8_fs_12_[6_12_24_48]/Checkpoint/seed/647/lr_0.0005_bs_8_lowloss.pth"
    check_point = torch.load(unet_path)
    check_point = {k: v for k, v in check_point.items() if not k.startswith("d")}
   # model.load_state_dict(check_point, strict=False)
    wandb.watch(model,criterion,log='all',log_freq=5) #this is freq of gradient recordings

    example_ct = 0
    batch_ct = 0
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_valid_score = [0.0]#in list so I can alter it in test function
    for epoch in tqdm(range(config.epochs)):

        avg_epoch_loss = 0.0
        start_time = time.time()
        counter = 0
        for _,(images, labels) in enumerate(loader):
            #start = time.perf_counter()
            points, point_labels, weak_unet_preds = gen_points_from_weak_unet_batch(images, labels, model,config.device,)
            #end = time.perf_counter()
           # print(f'point generation time: {end-start}s')
            if counter%1==0:
                plot = True


            else:
                plot = False

            plot = False

            counter+=1

            loss = prompt_train_batch(images,labels,points,point_labels,weak_unet_preds,model,optimizer,criterion,config,plot)

            avg_epoch_loss += loss
            example_ct += len(images)
            batch_ct +=1

            if ((batch_ct+1)%5)==0:
                prompt_train_log(loss,batch_ct,epoch)



        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)
        print(f'train time: {iteration_mins}m {iteration_secs}s')
        if (epoch)%3 == 0:
            test_results = prompt_test(model,test_loader,eval_criterion,config,best_valid_score,batch_ct)
            avg_epoch_loss/=len(loader)
            test_end_time = time.time()

            test_mins,test_secs = train_time(end_time,test_end_time)
            data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s Test Time: {test_mins}min {test_secs}s\n'
            data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
            data_str += test_results
            print(data_str)
    torch.save(model.state_dict(),config.final_path)
    save_model(config.final_path,"final_model")


def prompt_train_from_prev_model(model, loader, test_loader, criterion, eval_criterion, config, num_points=5,init_from_prev=False):

    if init_from_prev:
        unet_path = "/home/kebl6872/Desktop/new_data/REFUGE/test/unet_batch_lr_0.0003_bs_16_fs_12_[6_12_24_48]/Checkpoint/seed/279/lr_0.0003_bs_16_lowloss.pth"
        check_point = torch.load(unet_path)
        model.load_state_dict(check_point,strict=False)
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze promptImageCrossAttention and promptSelfAttention
        for param in model.promptImageCrossAttention.parameters():
            param.requires_grad = True
        for param in model.promptSelfAttention.parameters():
            param.requires_grad = True

        for param in model.d1.parameters():
            param.requires_grad = True
        for param in model.d2.parameters():
            param.requires_grad = True
        for param in model.d3.parameters():
            param.requires_grad = True
        for param in model.d4.parameters():
            param.requires_grad = True
        for param in model.outputs.parameters():
            param.requires_grad = True


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    wandb.watch(model, criterion, log='all', log_freq=50)  # this is freq of gradient recordings
    example_ct = 0
    batch_ct = 0

    #check initialised weights provide good baseline performance
    # for i, (images,labels) in enumerate(test_loader):
    #     if i > 5:
    #         break
    #     with torch.no_grad():
    #         initial_test(images,labels,model,eval_criterion)


    best_valid_score = [0.0]  # in list so I can alter it in test function
    for epoch in tqdm(range(config.epochs)):

        avg_epoch_loss = 0.0
        start_time = time.time()
        counter = 0
        for _, (images, labels) in enumerate(loader):
            prev_output = model(images.to(config.device,dtype=torch.float32), [], [], train_attention=False)
            points, point_labels = generate_points_batch(labels, prev_output,detach=True, num=1)

            for i in range(num_points):
                if counter % 1 == 0:
                    plot = True
                else:
                    plot = False


                plot = False
                counter += 1
                loss, points, point_labels,prev_output = prompt_train_batch_iterative(images, labels, points, point_labels,prev_output, model, optimizer,
                                                                criterion, plot)
                print(f'image {_}, point {i+1}: loss: {loss}')

            avg_epoch_loss += loss
            example_ct += len(images)
            batch_ct += 1

            if ((batch_ct + 1) % 4) == 0:
                prompt_train_log(loss, batch_ct, epoch)

        test_results = prompt_test(model, test_loader, eval_criterion, config, best_valid_score, batch_ct,num_points=num_points)
        avg_epoch_loss /= len(loader)
        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)
        data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
        data_str += test_results
        print(data_str)
    torch.save(model.state_dict(), config.final_path)
    save_model(config.final_path, "final_model")



def plot_output_and_prev(output,next_output, image, label, score, point_tuples,image_idx=0, detach=False):
        color_map = {0: 'red', 1: 'green', 2: 'blue'}
        if output.shape[1]==3:
            output = output.softmax(dim=1).argmax(dim=1).unsqueeze(1)
        image_np = image[image_idx, :, :, :].cpu().numpy().transpose(1, 2, 0)
        image_np = ((image_np * 127.5) + 127.5).astype(np.uint16)
        if not detach:
            output = output[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
            next_output = next_output[image_idx, :, :, :].cpu().numpy().transpose(1, 2, 0)
        else:
            output = output[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
            next_output = next_output[image_idx, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        label = label[image_idx, :, :, :].cpu().numpy().transpose(1, 2, 0)
        label = np.repeat(label, 3, 2)
        output = np.repeat(output, 3, 2)
        next_output = np.repeat(next_output, 3, 2)
        d = {0: 0, 1: 128, 2: 255}
        vfunc = np.vectorize(lambda x: d[x])

        # Apply the vectorized function to the array
        label = vfunc(label)
        output = vfunc(output)
        next_output = vfunc(next_output)
        rows = 1
        cols = 4

        # Create a figure with the specified size
        plt.figure(figsize=(15, 15))
        plt.subplot(rows, cols, 1)
        plt.imshow(image_np)
        plt.title("image")
        plt.axis('off')
        # Plot the mask in the even-numbered subplot
        plt.subplot(rows, cols, 2)
        plt.imshow(output)
        plt.title(f"prev_output + input pointsr")
        for y, x, val in point_tuples:  # point tuples stored in index e.g. ij = y,x
            print(f'({x},{y}): {val}')
            circle = plt.Circle((x, y), 2, color=color_map[val])
            plt.gca().add_patch(circle)
        plt.axis('off')
        plt.subplot(rows,cols,3)
        plt.imshow(next_output,)
        plt.title(f"current output {score}")
        for y, x, val in point_tuples:  # point tuples stored in index e.g. ij = y,x
            print(f'({x},{y}): {val}')
            circle = plt.Circle((x, y), 2, color=color_map[val])
            plt.gca().add_patch(circle)
        plt.axis("off")
        plt.subplot(rows, cols, 4)
        plt.imshow(label, )
        plt.title("ground truth")
        for y, x, val in point_tuples:  # point tuples stored in index e.g. ij = y,x
            print(f'({x},{y}): {val}')
            circle = plt.Circle((x, y), 2, color=color_map[val])
            plt.gca().add_patch(circle)
        plt.axis('off')

        # Show the plot
        plt.show()


def make_weakUNet(model_path):
    model = UNet(3, 3, 12, [6, 12, 24, 48], 'batch')

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def gen_components(indices_list,val):
    components = list()

    for indices in indices_list:
        map = np.zeros((512,512)).astype(np.uint8)
        if len(indices)==0:
            continue

        for i in range(indices.shape[0]):
            map[indices[i,0],indices[i,1]] = 1

        # generate component map
        (totalLabels, label_map, stats, centroids) = cv2.connectedComponentsWithStats(map, 8, cv2.CV_32S)

        # zip together component stats and their labels
        for (stat, componentLabel) in zip(stats[1:], list(range(1, totalLabels))):

            point_i,point_j = pick_rand(label_map,componentLabel)

            components.append([np.array([point_i,point_j]),stat[cv2.CC_STAT_AREA],val])

    res = sorted(components,key = lambda x: x[1],reverse=True)

    return [(x[0][0],x[0][1],x[2]) for x in res]


def gen_points_batch_from_model(y_true,output,num_points):
    output = output.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
    y_true = y_true.cpu().numpy().astype(int)
    output_o = torch.clone(output)
    output = output.detach().cpu().nump().astype(int)

    dc_misclass = np.argwhere(np.logical_and(y_true == 1, output == 2) == True)  # y_true indices are like [0,0,512,512]
    cd_misclass = np.argwhere(np.logical_and(y_true == 2, output == 1) == True)
    db_misclass = np.argwhere(np.logical_and(y_true == 1, output == 0) == True)
    cb_misclass = np.argwhere(np.logical_and(y_true == 2,output== 0) == True)
    bd_misclass = np.argwhere(np.logical_and(y_true == 0,output == 1) == True)
    bc_misclass = np.argwhere(np.logical_and(y_true == 0, output == 2) == True)


def gen_points_from_weak_unet(y_true, image,device,model, num_points, detach=False):
    model_paths = glob.glob(f'/home/kebl6872/Desktop/weakunet/Checkpoint/seed/**/*lowloss.pth', recursive=True)
    index = random.randint(len(model_paths),len(model_paths)*3-1)

    if index >= len(model_paths):
        prev_output = torch.from_numpy(np.zeros((1, 1, 512, 512))).to(device)  # need to make first input appear to havec come from model so it works w gen_points func
        points, point_labels = generate_points_batch(y_true, prev_output, num=1)
        point_input, point_label_input = torch.from_numpy(points).to(device, dtype=torch.float), torch.from_numpy(
            point_labels).to(device, dtype=torch.float)
        image = image.to(device, dtype=torch.float32)
        weak_unet_pred = model(image,point_input,point_label_input)

    else:
        weakUNet = make_weakUNet(model_paths[index])
        weakUNet = weakUNet.to(device)
        image = image.to(device,dtype=torch.float32)
        weak_unet_pred = weakUNet(image)

    weak_unet_pred = weak_unet_pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
    weak_unet_pred_o = torch.clone(weak_unet_pred)
    y_true = y_true.cpu().numpy().astype(int)
    weak_unet_pred = weak_unet_pred.cpu().numpy().astype(int)

    dc_misclass = np.argwhere(np.logical_and(y_true == 1, weak_unet_pred == 2) == True)[:,2:]  # y_true indices are like [0,0,512,512]
    cd_misclass = np.argwhere(np.logical_and(y_true==2,weak_unet_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    db_misclass = np.argwhere(np.logical_and(y_true==1,weak_unet_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    cb_misclass = np.argwhere(np.logical_and(y_true==2,weak_unet_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bd_misclass = np.argwhere(np.logical_and(y_true==0,weak_unet_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bc_misclass = np.argwhere(np.logical_and(y_true==0,weak_unet_pred==2)==True)[:,2:] # y_true indices are like [0,0,512,512]

    combined_results = list()
    type_register = {0 : 'background', 1 : 'disc', 2 : 'cup' }
    misclass_register = {0:[bd_misclass,bc_misclass],1:[db_misclass,dc_misclass],2:[cb_misclass,cd_misclass]}
    for i in range(len(misclass_register)):
        combined_results.append(gen_components(misclass_register[i],i))

    type_indexes = [0,0,0]
    res = list()
    for _ in range(num_points):
        class_type = random.randint(0,2)
        if type_indexes[class_type] >= len(combined_results[class_type]):

            correct_points = np.argwhere(y_true == class_type)[:,2:]
            l = list(range(correct_points.shape[0]))
            rand_point = correct_points[random.choice(l),:]
            res.append((rand_point[0],rand_point[1],class_type))

        else:
            res.append(combined_results[class_type][type_indexes[class_type]])
            type_indexes[class_type]+=1

    return res,weak_unet_pred_o

def gen_points_from_weak_unet_batch(images,y_true,model,device):

    num_points = random.randint(1,10)
    B = y_true.shape[0]

    points = np.zeros((B,num_points,2))
    point_labels = np.zeros((B,num_points,1))
    weak_unet_preds = list()
    for i in range(B):
        y_true_input = y_true[i, :, :, :]
        y_true_input = y_true_input[np.newaxis, :, :, :]  # need to add pseudo batch dimension to work w generate_points
        image_input = images[i,:,:,:]
        image_input = image_input[np.newaxis,:,:,:]
        gen_points,weak_unet_pred = gen_points_from_weak_unet(y_true_input,image_input,device,model,num_points)
        weak_unet_preds.append(weak_unet_pred)
        for j in range(num_points):

            points[i,j,0] = gen_points[j][0]
            points[i,j,1] = gen_points[j][1]
            point_labels[i,j,0] = gen_points[j][2]

    return points,point_labels,weak_unet_preds



def prompt_make(config):
    if config.dataset=="GS1":
        gs1 = True
    else:
        gs1 = False

    if config.dataset == "GAMMA":
        gamma=True
    else:
        gamma=False
    train1,train2 = get_data(train=False,transform=config.transform,),get_data(train=False,refuge_test=True,transform=config.transform)
    train = torch.utils.data.ConcatDataset([train1,train2])
    test1,test2 = get_data(train=True,gs1=True),get_data(train=True,gamma=True)
    test = torch.utils.data.ConcatDataset([test1,test2])
    eval_criterion = f1_valid_score
    train_loader = DataLoader(dataset=train,batch_size=config.batch_size,shuffle=True,)
    test_loader = DataLoader(dataset=test,batch_size=1,shuffle=False)
    criterion1 = DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
    criterion = NormalisedFocalLoss()
    diceFocal = combine_loss(criterion,criterion1,0.7)
    criterion2 = pointLoss(radius=10)
    pointCriterion = combine_point_loss(criterion2,diceFocal,alpha = 0.995,beta=200)

    model = PromptUNet(config.device,3,config.classes,config.base_c,dropout=0.1)


    return model,train_loader,test_loader,pointCriterion,eval_criterion
def prompt_model_pipeline(hyperparameters):
    with wandb.init(project="junk",config=hyperparameters):
        config = wandb.config

        model,train_loader,test_loader,criterion,eval_criterion = prompt_make(config)
        # print(model)
        model = model.to(device)
        prompt_train(model,train_loader,test_loader,criterion,eval_criterion,config)

    return model

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.autograd.set_detect_anomaly(True)

    wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')


    for _ in range(no_runs):
        config = dict(epochs=1000, classes=3, base_c = 12, kernels=[6,12,24,48], norm_name=norm_name,
                      batch_size=batch_size, learning_rate=lr, dataset="GAMMA",
                      architecture=model_name,seed=401,transform=True,device=device)
        config["seed"] = randint(201,400)
        seeding(config["seed"])


        data_save_path = f'/data_hd1/students/felix_cohen/Desktop/new_data/{config["dataset"]}/test/prompt{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{config["base_c"]}_[{"_".join(str(k) for k in config["kernels"])}]/'
        create_dir(data_save_path + f'Checkpoint/seed/{config["seed"]}')
        checkpoint_path_lowloss = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_lowloss.pth'
        checkpoint_path_final = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_final.pth'
        create_file(checkpoint_path_lowloss)
        create_file(checkpoint_path_final)
        config['low_loss_path']=checkpoint_path_lowloss
        config['final_path'] = checkpoint_path_final

        model = prompt_model_pipeline(config)


