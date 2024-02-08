import torch
import time
import random
import copy
# mask = torch.tensor([[[True,False],[False,False],[False,True]],
#                      [[False,False],[True,False],[False,True]],
#                      [[False,True],[False,False],[False,True]],
#                      [[False,False],[True,False],[False,True]],
#                      [[False,False],[False,True],[False,True]],
#                      [[False,False],[False,False],[True,True]],
#                      [[False,False],[True,False],[False,True]],
#                      [[False,False],[True,False],[False,True]]])
# mask = mask.unsqueeze(3)
# print(mask.shape)
# input = torch.tensor([[[[1,13,1,1],[2,14,1,1]],[[3,15,1,1],[4,16,1,1]],[[5,17,1,1],[6,18,1,1]]],[[[7,19,1,1],[8,20,1,1]],[[9,21,1,1],[10,22,1,1]],[[11,23,1,1],[12,24,1,1]]]])
# input = input.repeat(4,1,1,1)
# print(input.shape)
# masked_input = torch.masked_select(input,mask)
# masked_input = masked_input.reshape((8,2,4))
# print(mask[2,0,1,0])
# print(f'selected: {masked_input[2,:,:]}')
# print(f' og1: {input[2,0,1,:]}')
# print(f' og2: {input[2,2,1,:]}')
# output = torch.zeros((8,2,4)).long()
# #output[2,:,:] = masked_input[2,:,:]
# final_output = input.masked_scatter(mask,output)
# print(final_output[2,0,1,:])
# print(final_output[2,2,1,:])


def check_distance(self,feat_index, point_list, r):
    # diffs = torch.abs(feat_index[1:-1] - point_list[feat_index[0],:,:])
    point_list_indexed = torch.index_select(point_list, 0, feat_index[0].unsqueeze(0))

    # Use torch.unsqueeze to add a singleton dimension to feat_index[1:-1] and point_list_indexed
    feat_index_unsqueezed = torch.unsqueeze(feat_index[1:-1], 0)
    point_list_unsqueezed = torch.unsqueeze(point_list_indexed, 0)

    # Subtract the tensors and compute the absolute value
    diffs = torch.abs(feat_index_unsqueezed - point_list_unsqueezed)
    print(diffs.shape)
    dist = torch.max(diffs, dim=3)[0]
    print(f'dist: {dist}')

    dist_mask = torch.logical_not(torch.le(dist, r))
    print(f'dist mask: {dist_mask}')
    dist_mask = dist_mask.repeat(num_heads, 1, 1)


    return dist_mask

    # print(f'dist: {dist_mask.shape}')
    # #attention_mask[feat_index[0]*num_heads:(feat_index[0]+1)*num_heads,feat_index[-1],:] = dist_mask
    # attention_mask_indexed = torch.index_select(attention_mask, 0,feat_index[0].unsqueeze(0) * num_heads + torch.arange(num_heads))
    # attention_mask_indexed = torch.index_select(attention_mask_indexed, 1, feat_index[-1].unsqueeze(0))
    # print(f'indexed: {attention_mask_indexed.shape}')
    # # Assign dist_mask to the indexed tensor
    # # This will modify attention_mask in place
    # attention_mask_indexed[:] = dist_mask
check_distance_v = torch.vmap(check_distance,in_dims=(0,None,None))
B,H,W = 4,8,8
D = 10

images_input = torch.stack([torch.reshape(torch.arange(H*W*D),(H,W,D)) for _ in range(B)],dim=0)
print(images_input.shape)
print(images_input[0,3,3,:])
print(images_input[0,0,0,:])
pe = torch.stack([torch.reshape(torch.arange(H*W*D),(H,W,D)) for _ in range(B)],dim=0)
num_points = 2
radius = 6
L=((radius+1)**2)*num_points
add_max = L - 169
points = torch.zeros((B,num_points,2))
points[3,1,:] = torch.tensor([0,0])
points[3,0,:] = torch.tensor([0,5])
point_map = torch.zeros((B,H, W))
r_s = time.perf_counter()

for b in range(B):

    for point_idx in range(num_points):
        i, j = points[b, point_idx, 0].to(torch.int), points[b, point_idx, 1].to(torch.int)

        i_min = max(0, i - radius // 2)
        j_min = max(0, j - radius // 2)
        i_max = min(H, i + radius // 2)
        j_max = min(W, j + radius // 2)

        point_map[b, i_min:i_max + 1, j_min:j_max + 1].fill_(1)

        count = int(torch.sum(point_map[b, :, :]).item())


# additional_list = self.add_additional(point_map, H, W, b,add_max)
#
# idx = 0
#
# random.shuffle(additional_list[0])
# while count < L:
#     if len(additional_list[idx]) == 0:
#         idx += 1
#         random.shuffle(additional_list[idx])
#     i, j = additional_list[idx].pop()
#     point_map[b][i][j] = 1
#     count += 1
point_counts = torch.sum(point_map, dim=(1, 2))
print(f'point counts: {point_counts}')
point_map = point_map.bool().unsqueeze(3)


cumsum = torch.cumsum(point_counts, dim=0)
print(f'cumulative: {cumsum}')
# Create a list to store the padded tensors
padded = []
padded_pe = []

# Loop over each batch

masked_output = torch.masked_select(images_input, point_map)
masked_output = masked_output.pow(2)
masked_pe = torch.masked_select(pe, point_map)

padding_mask = torch.zeros((B, L))
for i in range(B):
    padding_mask[i, point_counts[i].to(torch.int):].fill_(1)
padding_mask = padding_mask.bool()

for i in range(B):
    # Get the output tensor for the current batch, shape: (points[i], D)
    # Use the cumulative sum to index the transformed tensor
    if i == 0:
        output = masked_output[:cumsum[i].to(torch.int) * D]
        pe_output = masked_pe[:cumsum[i].to(torch.int) * D]
    else:
        output = masked_output[cumsum[i - 1].to(torch.int) * D:cumsum[i].to(torch.int) * D]
        pe_output = masked_pe[cumsum[i - 1].to(torch.int) * D:cumsum[i].to(torch.int) * D]

    output = torch.reshape(output, (point_counts[i].to(torch.int), D))
    pe_output = torch.reshape(pe_output, (point_counts[i].to(torch.int), D))
    # Create a padding tensor of zeros, shape: (padding[i], D)
    pad = torch.zeros(L - point_counts[i].to(torch.int), D)

    # Concatenate the output and padding tensors along the second dimension, shape: (L, D)
    concat = torch.cat([output, pad], dim=0)
    concat_pe = torch.cat([pe_output, pad], dim=0)

    # Append the concatenated tensor to the list
    padded.append(concat)
    padded_pe.append(concat_pe)

# Stack the padded tensors along the first dimension, shape: (B, L, D)
masked_output = torch.stack(padded, dim=0)
masked_pe = torch.stack(padded_pe, dim=0)

print(f'masked_output shape: {masked_output.shape}')

print(padding_mask.shape)
masked_output = torch.masked_select(masked_output, torch.logical_not(padding_mask).unsqueeze(2))

images_input = images_input.to(torch.float)
images_input = images_input.masked_scatter(point_map, masked_output.to(torch.float))
num_heads = 2
attention_mask = torch.ones((num_heads*B,L,num_points))
selected_feats = torch.argwhere(point_map)
count = 0
prev = selected_feats[0, 0]
for i in range(selected_feats.size(0)):
    # If the 0th column value changes, reset the count
    if selected_feats[i, 0] != prev:
        count = 0
        prev = selected_feats[i, 0]
    # Replace the 4th column value with the count
    selected_feats[i, 3] = count
    # Increment the count
    count += 1


check = check_distance_v(selected_feats,points,self.radius//2)

for i in range(check.shape[0]):
    attention_mask[selected_feats[i][0] * num_heads:(selected_feats[i][0] + 1) * num_heads, selected_feats[i][-1], :] = check[i,:,0,:]


print(f'vectorised time: {t_e-t_s}')

# t_s = time.perf_counter()
# for b in range(B):
#     for i in range(H):
#         for j in range(W):
#              if point_map[b,i,j].item() == 1:
#                 for point_loc in range(points.shape[1]):
#                     p_i, p_j = points[b, point_loc, 0].to(torch.int), points[b, point_loc, 1].to(torch.int)
#                     if p_j == 2 or p_i == 11:
#                         print(f'i: {i} p_i: {p_i} j: {j} p_j: {p_j}')
#                     if abs(i-p_i) > radius//2 or abs(j-p_j) > radius//2:
#                         point_mask[b*num_heads:(b+1)*num_heads,feat_count,point_loc].fill_(1)
#                         if p_j == 2 or p_i == 11:
#                             print(point_mask[b*num_heads,feat_count,:])
#                 feat_count +=1
#     point_mask[b*num_heads:(b+1)*num_heads,feat_count:,:].fill_(1)
#     feat_count = 0
# t_e = time.perf_counter()
#
# print(f'normal time: {t_e-t_s}')
print(point_mask[6,:,:])
# print(images_input[0,3,3,:])
# print(images_input[3,0,0,:])
# print(images_input[3,0,5,:])
#



r_e = time.perf_counter()
print(f'reshape time: {r_e-r_s}')
