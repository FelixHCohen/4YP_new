import os
import random
import numpy as np
import torch
import cv2
import sys
import torch.nn as nn
from glob import glob
import random
import matplotlib.pyplot as plt
from pode import Contour, Polygon, divide, Requirement,Point
from pode import joined_constrained_delaunay_triangles

def seeding(seed):  # seeding the randomness
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_file(file):
    if not os.path.exists(file):
        open(file, "w")
    else:
        print(f"{file} Exists")


def train_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def segmentation_score(y_true, y_pred, num_classes):
    # returns confusion matrix (TP, FP, TN, FN) for each class, plus a combined class for class 1+2 (disc)
    if y_true.size() != y_pred.size():
        raise DimensionError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros((num_classes + 1, 5))

    for i in range(num_classes):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        tn = np.sum(np.logical_and(y_true != i, y_pred != i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        accuracy = (tp + tn)/(tp+fp+tn+fn+smooth)
        precision = tp/(tp+fp+smooth)
        recall = tp/(tp+fn+smooth)
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        IoU = tp/(tp+fp+fn+smooth)
        score_matrix[i] = np.array([IoU, f1, recall, precision, accuracy])
    # DISC
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    accuracy = (tp + tn) / (tp + fp + tn + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    IoU = tp / (tp + fp + fn + smooth)
    score_matrix[3] = np.array([IoU, f1, recall, precision, accuracy])

    return score_matrix

def f1_valid_two_classes(y_true,y_pred):
    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    # 0 == background class - will classify disk class as not 0 so I don't have to change the combined f1
    tp = np.sum(np.logical_and(y_true ==1, y_pred ==1))
    fp = np.sum(np.logical_and(y_true != 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true==1, y_pred==0))
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    return f1
def f1_valid_score(y_true, y_pred):
    if y_true.size() != y_pred.size():
        print(f' y true size: {y_true.size()} y_pred size: {y_pred.size()}')
        raise DimensionError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros(4)
    for i in range(3):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        score_matrix[i] = f1
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    score_matrix[3] = f1

    return score_matrix
def f1(tp,fp,fn):
    return 2*tp/(2*tp + (fp+fn))




def evaluate_centroids(y_true,map,indices,cup_scores,disc_scores,val,rank_by_area=True,cd=False,dc=False,db=False,bd=False,cb=False,bc=False):
    res = list()

    if len(indices) == 0: # no misclassifications therefore append an empty list
        return res

    for i in range(indices.shape[0]):
        map[indices[i, 0], indices[i, 1]] = 1

    (totalLabels, label_map, stats, centroids) = cv2.connectedComponentsWithStats(map, 8, cv2.CV_32S)



    for a, b, c in zip(stats[1:], centroids[1:], list(range(1, totalLabels))): #0th index corresponds to background component


        # centroid_i = round(b[1])
        # centroid_j = round(b[0])
        # string = ''
        # if cd:
        #     string = 'cd'
        #     delta_f1 = f1(cup_scores[0]+a[4],cup_scores[1],cup_scores[2]-a[4]) + f1(disc_scores[0],disc_scores[1]-a[4],disc_scores[2]) - f1(cup_scores[0],cup_scores[1],cup_scores[2]) - f1(disc_scores[0],disc_scores[1],disc_scores[2])
        #     val = 2
        #     if y_true[0,0,centroid_i,centroid_j] != 2:
        #         centroid_i,centroid_j = pick_rand(label_map,c)
        #     else:
        #         centroid_i, centroid_j = pick_rand(label_map, c)
        # elif dc:
        #     string = 'fc'
        #     delta_f1 = f1(cup_scores[0],cup_scores[1]-a[4],cup_scores[2]) + f1(disc_scores[0]+a[4],disc_scores[1],disc_scores[2]-a[4]) - f1(cup_scores[0],cup_scores[1],cup_scores[2]) - f1(disc_scores[0],disc_scores[1],disc_scores[2])
        #     val = 1
        #     if y_true[0,0,centroid_i,centroid_j] != 1:
        #         centroid_i,centroid_j = pick_rand(label_map,c)
        #     else:
        #         centroid_i, centroid_j = pick_rand(label_map, c)
        # elif db:
        #     string = 'db'
        #     #delta_f1 = f1(disc_scores[0]+a[4],disc_scores[1],disc_scores[2]-a[4]) - f1(disc_scores[0],disc_scores[1],disc_scores[2])
        #     val = 1
        #     if y_true[0,0,centroid_i,centroid_j] != 1:
        #         centroid_i,centroid_j = pick_rand(label_map,c)
        #     else:
        #         centroid_i, centroid_j = pick_rand(label_map, c)
        # elif bd:
        #     string = 'bd'
        #     #delta_f1 = f1(disc_scores[0], disc_scores[1]-a[4], disc_scores[2]) - f1(disc_scores[0], disc_scores[1], disc_scores[2])
        #     val = 0
        #     if y_true[0, 0, centroid_i, centroid_j] != 0:
        #         centroid_i, centroid_j = pick_rand(label_map, c)
        #     else:
        #         centroid_i, centroid_j = pick_rand(label_map, c)
        #
        # elif cb:
        #     string = 'cb'
        #     #delta_f1 = f1(cup_scores[0]+a[4], cup_scores[1], cup_scores[2] - a[4]) - f1(cup_scores[0], cup_scores[1], cup_scores[2])
        #     val = 2
        #     if y_true[0, 0, centroid_i, centroid_j] != 2:
        #         centroid_i, centroid_j = pick_rand(label_map, c)
        #     else:
        #         centroid_i, centroid_j = pick_rand(label_map, c)
        # elif bc:
        #     string = 'bc'
        #     #delta_f1 = f1(cup_scores[0], cup_scores[1]-a[4], cup_scores[2] ) - f1(cup_scores[0], cup_scores[1],cup_scores[2])
        #     val = 0
        #     #if y_true[0, 0, centroid_i, centroid_j] != 0:
        #         #centroid_i, centroid_j = pick_rand(label_map, c)
        #     #else:
        #     centroid_i, centroid_j = pick_rand(label_map, c)

        # if stats[c,cv2.CC_STAT_AREA] > 60:
        #     label_map_copy = label_map.copy()
        #     mask = (label_map == c)
        #     #
        #     # # Set the masked elements of label_map_copy to 1
        #     label_map_copy[mask]=1
        #     #
        #     # # Set the non-masked elements of label_map_copy to 0
        #     label_map_copy[~mask]=0
        #     plt.imshow(label_map_copy, cmap="gray")
        #     plt.title(f"misclassification {string}, area: {stats[c,cv2.CC_STAT_AREA]}")
        #
        #     plt.show()

        centroid_i,centroid_j = pick_rand(label_map,c)
        if not rank_by_area:
            res.append([np.array([centroid_i,centroid_j]),delta_f1,val,])
        else:
            res.append([np.array([centroid_i,centroid_j]),stats[c,cv2.CC_STAT_AREA],val,])

    return res

def pick_rand(map,label):
    indices = np.argwhere(map == label)
    l = list(range(indices.shape[0]))
    l_i = random.choice(l)
    return indices[l_i, :]



def generate_points(y_true,y_pred,num=1,detach=False):

    if y_pred.shape != (1,1,512,512):
        y_pred = y_pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)


    y_true = y_true.cpu().numpy().astype(int)

    if detach:
        y_pred = y_pred.detach().cpu().numpy().astype(int)
    else:
        y_pred = y_pred.cpu().numpy().astype(int)


    #
    # plt.subplot(1,2,1)
    # plt.imshow(y_pred[0,0,:,:],cmap='gray')
    # plt.title("y_pred")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(y_true[0,0,:,:], cmap='gray')
    # plt.title("y_true")
    # plt.show()
    combined_results = list()
    #each of the following misclassifications will affect avg f1 score differently
    maps = [np.zeros((512,512) ).astype(np.uint8) for _ in range(6)]
    dc_misclass = np.argwhere(np.logical_and(y_true==1,y_pred==2)==True)[:,2:] # y_true indices are like [0,0,512,512]
    cd_misclass = np.argwhere(np.logical_and(y_true==2,y_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    db_misclass = np.argwhere(np.logical_and(y_true==1,y_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]

    cb_misclass = np.argwhere(np.logical_and(y_true==2,y_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]

    bd_misclass = np.argwhere(np.logical_and(y_true==0,y_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bc_misclass = np.argwhere(np.logical_and(y_true==0,y_pred==2)==True)[:,2:] # y_true indices are like [0,0,512,512]

    disc_fp = cd_misclass.shape[0]+bd_misclass.shape[0]
    disc_fn = dc_misclass.shape[0] + db_misclass.shape[0]
    cup_fn = cd_misclass.shape[0]+cb_misclass.shape[0]
    cup_fp = dc_misclass.shape[0]+bc_misclass.shape[0]
    disc_tp = np.sum(np.logical_and(y_true==1,y_pred==1))
    cup_tp = np.sum(np.logical_and(y_true==2,y_pred==2))

    disc_scores = [disc_tp,disc_fp,disc_fn]
    cup_scores = [cup_tp,cup_fp,cup_fn]

    dc_centroids = evaluate_centroids(y_true,maps[0],dc_misclass,cup_scores,disc_scores,1,dc=True)
    combined_results.extend(dc_centroids)
    cd_centroids = evaluate_centroids(y_true,maps[1],cd_misclass,cup_scores,disc_scores,2,cd=True)
    combined_results.extend(cd_centroids)
    db_centroids = evaluate_centroids(y_true,maps[2],db_misclass,cup_scores,disc_scores,1,db=True)
    combined_results.extend(db_centroids)
    bd_centroids = evaluate_centroids(y_true,maps[3],bd_misclass,cup_scores,disc_scores,0,bd=True)
    combined_results.extend(bd_centroids)
    cb_centroids = evaluate_centroids(y_true,maps[4],cb_misclass,cup_scores,disc_scores,2,cb=True)
    combined_results.extend(cb_centroids)
    bc_centroids = evaluate_centroids(y_true,maps[5],bc_misclass,cup_scores,disc_scores,0,bc=True)
    combined_results.extend(bc_centroids)



    combined_results = sorted(combined_results, key=lambda x: x[1])



    return [(x[0][0],x[0][1],x[2],) for x in combined_results[-1*num:]] #returns p_i,p_j,p_label

def generate_points_batch(y_true,y_pred,num=1,detach=False):
    B = y_true.shape[0]
    points = np.zeros((B,num,2))
    point_labels = np.zeros((B,num,1))

    for i in range(B):
        y_true_input = y_true[i,:,:,:]
        y_true_input = y_true_input[np.newaxis,:,:,:]# need to add pseudo batch dimension to work w generate_points
        y_pred_input = y_pred[i, :, :, :]
        y_pred_input = y_pred_input[np.newaxis, :, :, :]  # need to add pseudo batch dimension to work w generate_points

        gen_points = generate_points(y_true_input,y_pred_input,num,detach=detach)

        for j in range(num):
            try:
                points[i,j,0] = gen_points[j][0]
                points[i,j,1] = gen_points[j][1]
                point_labels[i,j,0] = gen_points[j][2]
            except:
                print(f'gen points error {gen_points}')# points[i,j,0] = gen_points[j][0]
                points[i,j,1] = points[i,j-1,1]
                points[i, j, 0] = points[i, j - 1, 0]
                point_labels[i,j,0] = point_labels[i,j-1,0]

    return points,point_labels



def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)                # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


def norm(input: torch.tensor, norm_name: str):
    if norm_name == 'layer':
        normaliza = nn.LayerNorm(list(input.shape)[1:])
    elif norm_name == 'batch':
        normaliza = nn.BatchNorm2d(list(input.shape)[1])
    elif norm_name == 'instance':
        normaliza = nn.InstanceNorm2d(list(input.shape)[1])

    normaliza = normaliza.to(f'cuda:{input.get_device()}')

    output = normaliza(input)

    return output


def get_lr(step, lr):
    if step <= 100:
        lr_ = lr
    if step > 100:
        lr_ = lr + lr * np.cos(2 * np.pi * step / 100)

    return lr_


def choose_test_set(test_data_num):
    test_x = 'nothing'
    if test_data_num == 0:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
    elif test_data_num == 1:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_center_white_circle/*"))
    elif test_data_num == 2:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_corner_white_circle/*"))
    elif test_data_num == 3:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_edge_white_circle/*"))
    elif test_data_num == 4:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.1/*"))
    elif test_data_num == 5:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.2/*"))
    elif test_data_num == 6:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.3/*"))
    elif test_data_num == 7:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.4/*"))
    elif test_data_num == 8:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.5/*"))
    elif test_data_num == 9:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.1/*"))
    elif test_data_num == 10:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.2/*"))
    elif test_data_num == 11:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.3/*"))
    elif test_data_num == 12:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.4/*"))
    elif test_data_num == 13:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.5/*"))
    elif test_data_num == 14:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.1/*"))
    elif test_data_num == 15:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.2/*"))
    elif test_data_num == 16:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.3/*"))
    elif test_data_num == 17:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.4/*"))
    elif test_data_num == 18:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.5/*"))
    elif test_data_num == 19:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_match/*"))
    elif test_data_num == 20:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/greenlight/*"))
    return test_x
