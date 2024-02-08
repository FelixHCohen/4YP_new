from PromptUNetTrain import *


def prompt_test_make(config,path,device):

    model = PromptUNet(config['device'],3,config['classes'],config['base_c'],kernels=config['kernels'],attention_kernels=config['attention_kernels'],d_model=config['d_model'],dropout=0.1,batch_norm=config['batch_norm'])
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint,strict=False)
    model = model.to(device)
    model.eval()

    if config['testset'] == 'GAMMA':
        train = get_data(train=True, return_path=False, gamma=True, transform=False, )
        test = get_data(train=False, return_path=False, gamma=True, transform=False, )
        if config['dataset'] != 'GAMMA':
            total = torch.utils.data.ConcatDataset([train, test])
        else:
            total = test

    elif config["testset"] == "GS1":
        train = get_data(train=True, return_path=False, gs1=True, transform=False, )
        test = get_data(train=False, return_path=False, gs1=True, transform=False, )
        if config['dataset'] != 'GS1':
            total = torch.utils.data.ConcatDataset([train, test])
        else:

            total = test

    else:
        total = get_data(train=True, return_path=False, transform=False, )

    train_loader = DataLoader(dataset=total, batch_size=1, shuffle=True, )

    criterion = f1_valid_score

    return model, train_loader,criterion

if __name__ == '__main__':
    device = (torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')

    with wandb.init(project="ARC-TESTING", config={}, dir='/data/engs-mlmi1/kebl6872/wandb'):
        for testset in ['GAMMA','GS1','REFUGE TRAIN']:
            print(testset)
            print('-----')
            model_str = "/data/engs-mlmi1/kebl6872/data/models/PromptUNet_lr_0.0005_bs_12_fs_12_[6_12_24_48]_3_2_5_5]/Checkpoint/seed/987/lr_0.0005_bs_12_lowloss.pth"

            config = dict(epochs=1000, classes=3, base_c = 12, kernels=[6,12,24,48],attention_kernels = [3,2,5,5],d_model=64,
                              batch_size=1, dataset="REFUGE_VAL_TEST",
                              testset=testset,seed=401,transform=True,device=device,batch_norm=False)
            model, loader, criterion = prompt_test_make(config,model_str,device)
            num_points = 20
            with torch.no_grad():

                val_scores = np.zeros(num_points)
                f1_score_record = np.zeros((4,num_points))
                total = 0
                for _,(images,labels) in enumerate(loader):
                    images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                    prev_output = torch.from_numpy(np.zeros((1,1,512,512))).to(device)# need to make first input appear to havec come from model so it works w gen_points func
                    points, point_labels = generate_points_batch(labels,prev_output ,num=1)
                    #prev_output = prev_output.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
                    # above line was for when prev_output came out of model
                    for i in range(num_points):

                        point_input, point_label_input = torch.from_numpy(points).to(device,dtype=torch.float32), torch.from_numpy(point_labels).to(device, dtype=torch.float32)
                        outputs = model(images, point_input, point_label_input,train_attention=True)

                        new_points, new_point_labels = generate_points_batch(labels, outputs)

                        outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)

                        score = criterion(outputs,labels)


                        val_scores[i] += score[1].item() / 2 + score[2].item() / 2

                        f1_score_record[:,i] += score
                        plot =False
                        if plot:

                            image_idx = random.randint(0, images.shape[0] - 1)
                            point_tuples = [(i, j, val[0]) for (i, j), val in
                                            zip(points[image_idx, :, :], point_labels[image_idx, :, :])]
                            print(f'point_tuples: {point_tuples}')
                            plot_output_and_prev(prev_output[image_idx,:,:,:].unsqueeze(dim=0).detach(),outputs, images, labels, score,
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
            print(return_str)


