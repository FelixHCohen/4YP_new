import torch
from torch import nn
from PromptUNet.PromptEncoder import *
import random
class ResidualCrossConnection(ResidualConnection):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
            super().__init__(d_model,dropout)

    def forward(self,x,y,sublayer,attn_mask=False):
        output = self.dropout(sublayer(x,y,attn_mask))
        return self.norm(x + output)


class ResidualBatchCrossConnection(nn.Module):
    def __init__(self,d_model,dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,y,sublayer):
        output = self.dropout(sublayer(x,y))
        output = x + output
        output = output.transpose(1,2)
        output = self.norm(output)
        output = output.transpose(1,2)
        return output

class ResidualBatchNormPosConnection(nn.Module):
    def __init__(self,d_model,dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,pos,add_pos,sublayer):
        output = self.dropout(sublayer(x))

        output = x+output
        output = output.transpose(1,2)
        output = self.norm(output)
        output = output.transpose(1,2)
        if add_pos:
            output += pos
        return output
class ResidualPosConnection(ResidualConnection):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__(d_model, dropout)

    def forward(self, x, pos,add_pos, sublayer):
        output = self.dropout(sublayer(x))
        if add_pos:
            output += pos
        return self.norm(x + output)
class Embeddings(nn.Module):

    def __init__(self,d_model, size,device):
        super().__init__()
        self.d_model = d_model
        self.num_labels = size[0]*size[1]
        self.embedding = nn.Embedding(self.num_labels,d_model,device=device)

    def forward(self,x):
        return self.embedding(x)


class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()

        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.attn_layer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)



    def forward(self, q_tensors,kv_tensors,attn_mask=False):  # x_shape = B,L+1,d_model

        Q = self.w_q(q_tensors)
        K = self.w_k(kv_tensors)
        V = self.w_v(kv_tensors)
        if torch.is_tensor(attn_mask):
            attn_output = self.attn_layer(Q,K,V,need_weights=False,attn_mask=attn_mask)
        else:
            attn_output = self.attn_layer(Q, K, V, need_weights=False)

        return attn_output[0]

class CrossAttentionBlockBatch(nn.Module): #uses batch norm rather than layer norm
    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.CrossAttention1 = MultiHeadCrossAttentionLayer(d_model, num_heads, dropout)
        self.FFN = FFN(d_model, dropout)
        self.res_connection1 = ResidualCrossConnection(d_model, dropout)
        self.res_connection2 = ResidualConnection(d_model, dropout)
        self.CrossAttention2 = MultiHeadCrossAttentionLayer(d_model, num_heads, dropout)
        self.FFN2 = FFN(d_model, dropout)
        self.res_connection3 = ResidualBatchCrossConnection(d_model, dropout)
        self.res_connection4 = ResidualBatchNormPosConnection(d_model, dropout)

    def forward(self, images, prompts_input, original_prompts, pos_encodings, add_pos=True):
        images = self.res_connection3(images, prompts_input, self.CrossAttention2)
        images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)
        prompts_input = self.res_connection1(prompts_input, images, self.CrossAttention1)
        prompts_input = self.res_connection2(prompts_input, self.FFN)
        prompts_output = prompts_input + original_prompts
        return images, prompts_output
class CrossAttentionBlock(nn.Module):

    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.CrossAttention1 = MultiHeadCrossAttentionLayer(d_model,num_heads,dropout)
        self.FFN = FFN(d_model,dropout)
        self.res_connection1 = ResidualCrossConnection(d_model,dropout)
        self.res_connection2 = ResidualConnection(d_model,dropout)
        self.CrossAttention2 = MultiHeadCrossAttentionLayer(d_model,num_heads,dropout)
        self.FFN2 = FFN(d_model, dropout)
        self.res_connection3 = ResidualCrossConnection(d_model, dropout)
        self.res_connection4 = ResidualPosConnection(d_model, dropout)

    def forward(self,images,prompts_input,original_prompts,pos_encodings,add_pos=True,attn_mask=False):
        # if torch.is_tensor(attn_mask):
        #     prompt_attn_mask = torch.transpose(attn_mask,1,2)
        # else:
        #     prompt_attn_mask = False
        # prompts_input = self.res_connection1(prompts_input, images, self.CrossAttention1,prompt_attn_mask)  # if using masked encoder need to hide empty image K-V pairs
        # prompts_input = self.res_connection2(prompts_input, self.FFN)
        # images = self.res_connection3(images, prompts_input, self.CrossAttention2,attn_mask)
        # images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)

        images = self.res_connection3(images, prompts_input, self.CrossAttention2)
        images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)
        prompts_input = self.res_connection1(prompts_input, images, self.CrossAttention1, attn_mask)
        prompts_input = self.res_connection2(prompts_input, self.FFN)


        prompts_output = prompts_input + original_prompts
        return images,prompts_output

class ImageEncoder(nn.Module):

    def __init__(self,device,embeddings,d_model=128,d_image_in=576, num_heads=8,num_blocks=6, dropout=0.1,batch=False):
        super().__init__()
        self.device = device
        self.d_model=d_model
        self.image_feature_embedding = nn.Linear(d_image_in,d_model)
        self.embedding_to_feature = nn.Linear(d_model,d_image_in)
        self.embeddings = embeddings
        self.batch = batch
        if batch:
            self.cross_attns = nn.ModuleList([CrossAttentionBlockBatch(d_model,num_heads,dropout) for _ in range(num_blocks)])
        else:
            self.cross_attns = nn.ModuleList([CrossAttentionBlock(d_model,num_heads,dropout) for _ in range(num_blocks)])
            self.ln = nn.LayerNorm(d_model)



    def pos_emb_grid(self,B, H, W):
        grid = torch.ones((H, W), device=self.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / H
        x_embed = x_embed / W

        pe = self.embeddings._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        pe = pe.permute(2, 0, 1)  # d_model x H x W

        pe = pe.unsqueeze(0).repeat(B, 1, 1, 1)  # shape Bxd_modelxHxW

        return pe



    def forward(self,images_input,prompts,original_prompts,):
        B,D,H,W = images_input.shape
        #print(f'b: {B} D {D} H {H} W {W}')

        pe = self.pos_emb_grid(B,H,W)

        #print(f'pe shape: {pe.shape}')

        images_input = torch.permute(images_input,(0,2,3,1)) # rearange to BxHxWxD so when we flatten vectors are sorted by batch

        images_input = self.image_feature_embedding(images_input)
        #images_input = images_input.reshape(B,H,W,self.d_model)

        images_input = torch.permute(images_input,(0,3,1,2)) # now shape B,d_model,H,W

        #images = images_input + pe # += is an in place alteration and messes up backprop

        images_input = images_input.view(B,self.d_model,-1)

        images_input = images_input.permute(0,2,1)
        pe = pe.view(B,self.d_model,-1)
        pe = pe.permute(0,2,1)
        images = images_input + pe

        if not self.batch:
            images = self.ln(images)  # trying layernorm before first input into cross attention

        add_pos = True
        #print(f'input shape: {images.shape}')

        for i in range(len(self.cross_attns)):
            if i == len(self.cross_attns)-1:
                add_pos = False
            images,prompts = self.cross_attns[i](images,prompts,original_prompts,pe,add_pos)

        images = self.embedding_to_feature(images)
        images = torch.reshape(images,(B,D,H,W))

        return images,prompts



class _maskedImageEncoder(ImageEncoder):
    def __init__(self, device, embeddings, d_model=128, d_image_in=576, num_heads=8, num_blocks=6, dropout=0.1,radius=10,box=False):
        super().__init__(device,embeddings,d_model,d_image_in,num_heads,num_blocks,dropout)
        self.num_heads = num_heads
        self.radius = radius
        self.box = box

    def add_additional(self, point_map, H, W, b, max): #BFS algorithm that finds N closest points to +ve islands on point map ordered by manhattan distance
        matrix = torch.clone(point_map[b, :, :]) # will use this to keep track of visited points
        additional_list = []
        queue = torch.argwhere(matrix).tolist() # think torch where functions support parallelisation therefore faster than for loops
        # Define the four directions to move in the matrix
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        add_count = 0

        while queue and add_count < max: # max amount of overlapped squares = num_points*(radius + 1)**2 - (radius//2 +1)**2 (all points are in a corner)
            # Get the size of the current level of the queue
            size = len(queue)


            mask = torch.zeros((H + 2, W + 2)) # mask has to be larger than matrix to account for invalid direction steps e.g. right corner pixel + [0,1]
            # Loop through the current level of the queue
            for _ in range(size):
                # Pop the first element
                x, y = queue.pop()
                # place x and y positions correctly in larger mask [0,0]->[1,1] etc.
                x += 1
                y += 1

                # Loop through the four directions
                for dx, dy in directions:
                    # Calculate the new candidates for distance x away from islands
                    nx = x + dx
                    ny = y + dy
                    mask[nx, ny] = 1
            # create final mask that is intersection of unvisited points and new candidate points we are specified by directions - these form all points distance x from original 'point islands'
            final_mask = torch.logical_and(mask[1:-1, 1:-1], torch.logical_not(matrix))
            curr_distance = torch.argwhere(final_mask).tolist()
            queue = torch.argwhere(final_mask).tolist()
            matrix = torch.where(final_mask.bool(), torch.tensor(1.0), matrix)
            add_count += len(curr_distance)
            additional_list.append(curr_distance)

        return additional_list # returns list of sets indices in which index 0 is all coordinates distance 1 from point islands, index 1 is all points distance 2 etc. (manhattan distance)

    def forward(self, images_input, prompts, original_prompts, points,):
        B, D, H, W = images_input.shape

        pe = self.pos_emb_grid(B,H,W)


        points /= (512/H) # assuming input image is square, matches point to feature resolution ( .to(torch.int) will truncate decimal values later)

        num_points = points.shape[1]
        L = num_points*(self.radius+1)**2
        add_max = L - (self.radius//2 + 1)**2


        images_input = torch.permute(images_input,(0, 2, 3, 1))  # rearange to BxHxWxD
        #images_input = self.image_feature_embedding(images_input)

        point_map = torch.zeros((B,H, W))

        for b in range(B):

                for point_idx in range(num_points):
                    i,j = points[b,point_idx,0].to(torch.int),points[b,point_idx,1].to(torch.int)

                    i_min = max(0,i- self.radius//2)
                    j_min = max(0,j- self.radius//2)
                    i_max = min(H,i+ self.radius//2)
                    j_max = min(W,j+ self.radius//2)

                    point_map[b,i_min:i_max+1,j_min:j_max+1].fill_(1)
                  #  count = int(torch.sum(point_map[b, :, :]).item())


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


        point_map = point_map.bool().to(self.device).unsqueeze(3)
        pe = pe.permute(0,2,3,1)

        cumsum = torch.cumsum(point_counts, dim=0)

        # Create a list to store the padded tensors
        padded = []
        padded_pe = []

        # Loop over each batch
        masked_output = torch.masked_select(images_input, point_map)
        masked_pe = torch.masked_select(pe, point_map)

        padding_mask = torch.zeros((B,L))
        for i in range(B):
            padding_mask[i,point_counts[i].to(torch.int):].fill_(1)
        padding_mask = padding_mask.bool().to(self.device)

        for i in range(B):
            # Get the output tensor for the current batch, shape: (points[i], D)
            # Use the cumulative sum to index the transformed tensor
            if i == 0:
                output = masked_output[:cumsum[i].to(torch.int)*D]
                pe_output = masked_pe[:cumsum[i].to(torch.int)*self.d_model]
            else:
                output = masked_output[cumsum[i - 1].to(torch.int)*D:cumsum[i].to(torch.int)*D]
                pe_output = masked_pe[cumsum[i - 1].to(torch.int)*self.d_model:cumsum[i].to(torch.int)*self.d_model]

            output = torch.reshape(output,(point_counts[i].to(torch.int),D))
            pe_output = torch.reshape(pe_output, (point_counts[i].to(torch.int), self.d_model))
            # Create a padding tensor of zeros, shape: (padding[i], D)
            pad = torch.zeros(L - point_counts[i].to(torch.int), D)

            # Concatenate the output and padding tensors along the second dimension, shape: (L, D)
            concat = torch.cat([output, pad], dim=0)
            concat_pe = torch.cat([pe_output,pad[:,self.d_model]],dim=0)

            # Append the concatenated tensor to the list
            padded.append(concat)
            padded_pe.append(concat_pe)

        # Stack the padded tensors along the first dimension, shape: (B, L, D)
        masked_output = torch.stack(padded, dim=0).to(self.device)
        masked_pe = torch.stack(padded_pe,dim=0).to(self.device)



        masked_output = self.image_feature_embedding(masked_output)


        masked_output = masked_output + masked_pe

        masked_output = self.ln(masked_output)  # trying layernorm before first input into cross attention

        add_pos = True
        for i in range(len(self.cross_attns)):

            if i == len(self.cross_attns) - 1:
                add_pos = False

            masked_output, prompts = self.cross_attns[i](masked_output, prompts, original_prompts, masked_pe, add_pos,padding_mask)



        masked_output = self.embedding_to_feature(masked_output)
        masked_output = torch.masked_select(masked_output,torch.logical_not(padding_mask).unsqueeze(2))
        images_input = images_input.masked_scatter(point_map,masked_output)

        images_input = images_input.permute(0,3,1,2)

        return images_input,prompts


class maskedImageEncoder(ImageEncoder):
    def __init__(self, device, embeddings, d_model=128, d_image_in=576, num_heads=8, num_blocks=6, dropout=0.1,
                 radius=10, box=False):
        super().__init__(device, embeddings, d_model, d_image_in, num_heads, num_blocks, dropout)
        self.num_heads = num_heads
        self.radius = radius
        self.box = box

    def check_distance(self, feat_index, point_list, r):
        # diffs = torch.abs(feat_index[1:-1] - point_list[feat_index[0],:,:])
        point_list_indexed = torch.index_select(point_list, 0, feat_index[0].unsqueeze(0))

        # Use torch.unsqueeze to add a singleton dimension to feat_index[1:-1] and point_list_indexed
        feat_index_unsqueezed = torch.unsqueeze(feat_index[1:-1], 0)
        point_list_unsqueezed = torch.unsqueeze(point_list_indexed, 0)

        # Subtract the tensors and compute the absolute value
        diffs = torch.abs(feat_index_unsqueezed - point_list_unsqueezed)

        dist = torch.max(diffs, dim=3)[0]

        dist_mask = torch.logical_not(torch.le(dist, r))

        dist_mask = dist_mask.repeat(self.num_heads, 1, 1)

        return dist_mask

    def forward(self, images_input, prompts, original_prompts, points, mask=None):
        B, D, H, W = images_input.shape

        pe = self.pos_emb_grid(B, H, W)

        check_distance_v = torch.vmap(self.check_distance, in_dims=(0, None, None))

        points /= (512 / H)  # assuming input image is square, matches point to feature resolution ( .to(torch.int) will truncate decimal values later)

        num_points = points.shape[1]
        L = num_points * (self.radius + 1) ** 2
        add_max = L - (self.radius // 2 + 1) ** 2

        images_input = torch.permute(images_input, (0, 2, 3, 1))  # rearange to BxHxWxD
        # images_input = self.image_feature_embedding(images_input)

        point_map = torch.zeros((B, H, W))

        for b in range(B):

            for point_idx in range(num_points):
                i, j = points[b, point_idx, 0].to(torch.int), points[b, point_idx, 1].to(torch.int)

                i_min = max(0, i - self.radius // 2)
                j_min = max(0, j - self.radius // 2)
                i_max = min(H, i + self.radius // 2)
                j_max = min(W, j + self.radius // 2)

                point_map[b, i_min:i_max + 1, j_min:j_max + 1].fill_(1)
            #  count = int(torch.sum(point_map[b, :, :]).item())

        point_counts = torch.sum(point_map, dim=(1, 2))

        point_map = point_map.bool().to(self.device).unsqueeze(3)
        pe = pe.permute(0, 2, 3, 1)

        cumsum = torch.cumsum(point_counts, dim=0)

        # Create a list to store the padded tensors
        padded = []
        padded_pe = []

        # Loop over each batch
        #move selected vectors to new tensor so we dont apply linear projection to cross attention dimension to all vectors needlessly
        masked_output = torch.masked_select(images_input, point_map)
        masked_pe = torch.masked_select(pe, point_map)

        padding_mask = torch.zeros((B, L))
        for i in range(B):
            padding_mask[i, point_counts[i].to(torch.int):].fill_(1)
        padding_mask = padding_mask.bool().to(self.device)

        for i in range(B):
            # Get the output tensor for the current batch, shape: (points[i], D)
            # Use the cumulative sum to index the transformed tensor
            if i == 0:
                output = masked_output[:cumsum[i].to(torch.int) * D]
                pe_output = masked_pe[:cumsum[i].to(torch.int) * self.d_model]
            else:
                output = masked_output[cumsum[i - 1].to(torch.int) * D:cumsum[i].to(torch.int) * D]
                pe_output = masked_pe[cumsum[i - 1].to(torch.int) * self.d_model:cumsum[i].to(torch.int) * self.d_model]

            output = torch.reshape(output, (point_counts[i].to(torch.int), D))
            pe_output = torch.reshape(pe_output, (point_counts[i].to(torch.int), self.d_model))
            # Create a padding tensor of zeros, shape: (padding[i], D)

            pad = torch.zeros(L - point_counts[i].to(torch.int), D).to(self.device)

            # Concatenate the output and padding tensors along the second dimension, shape: (L, D)
            concat = torch.cat([output, pad], dim=0)
            concat_pe = torch.cat([pe_output, pad[:, :self.d_model]], dim=0)

            # Append the concatenated tensor to the list
            padded.append(concat)
            padded_pe.append(concat_pe)

        # Stack the padded tensors along the first dimension, shape: (B, L, D)
        masked_output = torch.stack(padded, dim=0).to(self.device)
        masked_pe = torch.stack(padded_pe, dim=0).to(self.device)

        masked_output = self.image_feature_embedding(masked_output)

        masked_output = masked_output + masked_pe

        masked_output = self.ln(masked_output)  # trying layernorm before first input into cross attention

        if not torch.is_tensor(mask):
            attention_mask = torch.ones((self.num_heads * B, L, num_points))
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

            check = check_distance_v(selected_feats, points, self.radius // 2)

            for i in range(check.shape[0]):
                attention_mask[selected_feats[i][0] * self.num_heads:(selected_feats[i][0] + 1) * self.num_heads,
                selected_feats[i][-1], :] = check[i, :, 0, :]

            # attention_mask = torch.zeros((self.num_heads*B,L,num_points))

            # feat_count = 0
            # for b in range(B):
            #     for i in range(H):
            #         for j in range(W):
            #             if point_map[b,i,j].item() == True:
            #                 for point_loc in range(points.shape[1]):
            #                     p_i, p_j = points[b, point_loc, 0].to(torch.int), points[b, point_loc, 1].to(torch.int)
            #                     if abs(i-p_i) > self.radius//2 or abs(j-p_j) > self.radius//2:
            #                         attention_mask[b*self.num_heads:(b+1)*self.num_heads,feat_count,point_loc].fill_(1)
            #                 feat_count +=1
            #     attention_mask[b*self.num_heads:(b+1)*self.num_heads,feat_count:,:].fill_(1)
            #     feat_count = 0

            attention_mask = attention_mask.transpose_(1,2).bool().to(self.device)
        else:
            attention_mask = mask

        add_pos = True
        for i in range(len(self.cross_attns)):

            if i == len(
                    self.cross_attns) - 1:  # no need to add positional encoding after final cross attention layer -> positional encodings of next set of patches are different resolution
                add_pos = False

            masked_output, prompts = self.cross_attns[i](masked_output, prompts, original_prompts, masked_pe, add_pos,
                                                         attention_mask)

        masked_output = self.embedding_to_feature(masked_output)
        masked_output = torch.masked_select(masked_output, torch.logical_not(padding_mask).unsqueeze(2))
        images_input = images_input.masked_scatter(point_map, masked_output)

        images_input = images_input.permute(0, 3, 1, 2)

        return images_input, prompts if torch.is_tensor(mask) else images_input,prompts,attention_mask
class maskedBoxImageEncoder(ImageEncoder):
    def __init__(self, device, embeddings, d_model=128, d_image_in=576, num_heads=8, num_blocks=6, dropout=0.1,
                 radius=10, box=False):
        super().__init__(device, embeddings, d_model, d_image_in, num_heads, num_blocks, dropout)
        self.num_heads = num_heads
        self.radius = radius
        self.box = box

    def add_additional(self, point_map, H, W, b,
                       max):  # BFS algorithm that finds N closest points to +ve islands on point map ordered by manhattan distance
        matrix = torch.clone(point_map[b, :, :])  # will use this to keep track of visited points
        additional_list = []
        queue = torch.argwhere(
            matrix).tolist()  # think torch where functions support parallelisation therefore faster than for loops
        # Define the four directions to move in the matrix
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        add_count = 0

        while queue and add_count < max:  # max amount of overlapped squares = num_points*(radius + 1)**2 - (radius//2 +1)**2 (all points are in a corner)
            # Get the size of the current level of the queue
            size = len(queue)

            mask = torch.zeros((H + 2,W + 2))  # mask has to be larger than matrix to account for invalid direction steps e.g. right corner pixel + [0,1]
            # Loop through the current level of the queue
            for _ in range(size):
                # Pop the first element
                x, y = queue.pop()
                # place x and y positions correctly in larger mask [0,0]->[1,1] etc.
                x += 1
                y += 1

                # Loop through the four directions
                for dx, dy in directions:
                    # Calculate the new candidates for distance x away from islands
                    nx = x + dx
                    ny = y + dy
                    mask[nx, ny] = 1
            # create final mask that is intersection of unvisited points and new candidate points we are specified by directions - these form all points distance x from original 'point islands'
            final_mask = torch.logical_and(mask[1:-1, 1:-1], torch.logical_not(matrix))
            curr_distance = torch.argwhere(final_mask).tolist()
            queue = torch.argwhere(final_mask).tolist()
            matrix = torch.where(final_mask.bool(), torch.tensor(1.0), matrix)
            add_count += len(curr_distance)
            additional_list.append(curr_distance)

        return additional_list  # returns list of sets indices in which index 0 is all coordinates distance 1 from point islands, index 1 is all points distance 2 etc. (manhattan distance)

    def forward(self, images_input, prompts, original_prompts, points, ):
        B, D, H, W = images_input.shape

        pe = self.pos_emb_grid(B, H, W)

        points /= (512 / H)  # assuming input image is square, matches point to feature resolution ( .to(torch.int) will truncate decimal values later)

        num_points = points.shape[1]
        L = num_points * (self.radius + 1) ** 2
        add_max = L - (self.radius // 2 + 1) ** 2

        if self.box:
            L = (H * W) // 8
            point_mask = torch.zeros((B, num_points,))

        masked_output = torch.zeros((B, L, D)).to(self.device)
        masked_pe = torch.zeros((B, L, self.d_model)).to(self.device)
        images_input = torch.permute(images_input, (0, 2, 3, 1))  # rearange to BxHxWxD
        # images_input = self.image_feature_embedding(images_input)

        point_map = torch.zeros((B, H, W))

        for b in range(B):
            if not self.box:
                for point_idx in range(num_points):
                    i, j = points[b, point_idx, 0].to(torch.int), points[b, point_idx, 1].to(torch.int)

                    i_min = max(0, i - self.radius // 2)
                    j_min = max(0, j - self.radius // 2)
                    i_max = min(H, i + self.radius // 2)
                    j_max = min(W, j + self.radius // 2)

                    point_map[b, i_min:i_max + 1, j_min:j_max + 1].fill_(1)
                    count = int(torch.sum(point_map[b, :, :]).item())
            else:
                i_left, j_left = points[b, 0, 0].to(torch.int), points[b, 0, 1].to(torch.int)
                i_right, j_right = points[b, 1, 0].to(torch.int), points[b, 1, 1].to(torch.int),

                point_map[b, i_left:i_right + 1, j_left:j_right + 1].fill_(1)
                count = int(torch.sum(point_map[b, :, :]).item())
                add_max = L - count

                for point_idx in range(num_points):
                    i, j = points[b, point_idx, 0].to(torch.int), points[b, point_idx, 1].to(torch.int)
                    if i < 0.9 * i_left or i > 1.1 * i_right or j < 0.9 * j_left or j > 0.9 * j_right:
                        point_mask[b, point_idx] = 1

                point_mask = point_mask.bool()

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

        point_map = point_map.bool().to(self.device).unsqueeze(3)
        pe = pe.permute(0, 2, 3, 1)

        cumsum = torch.cumsum(point_counts, dim=0)

        # Create a list to store the padded tensors
        padded = []
        padded_pe = []

        # Loop over each batch
        masked_output = torch.masked_select(images_input, point_map)
        masked_pe = torch.masked_select(pe, point_map)

        padding_mask = torch.zeros((B, L))
        for i in range(B):
            padding_mask[i, point_counts[i].to(torch.int):].fill_(1)

        for i in range(B):
            # Get the output tensor for the current batch, shape: (points[i], D)
            # Use the cumulative sum to index the transformed tensor
            if i == 0:
                output = masked_output[:cumsum[i].to(torch.int) * D]
                pe_output = masked_pe[:cumsum[i].to(torch.int) * self.d_model]
            else:
                output = masked_output[cumsum[i - 1].to(torch.int) * D:cumsum[i].to(torch.int) * D]
                pe_output = masked_pe[cumsum[i - 1].to(torch.int) * D:cumsum[i].to(torch.int) * self.d_model]

            output = torch.reshape(output, (point_counts[i].to(torch.int), D))
            pe_output = torch.reshape(pe_output, (point_counts[i].to(torch.int), self.d_model))
            # Create a padding tensor of zeros, shape: (padding[i], D)
            pad = torch.zeros(L - point_counts[i].to(torch.int), D).to(self.device)

            # Concatenate the output and padding tensors along the second dimension, shape: (L, D)
            concat = torch.cat([output, pad], dim=0)
            concat_pe = torch.cat([pe_output, pad[:,:self.d_model]], dim=0)

            # Append the concatenated tensor to the list
            padded.append(concat)
            padded_pe.append(concat_pe)

        # Stack the padded tensors along the first dimension, shape: (B, L, D)
        masked_output = torch.stack(padded, dim=0)
        masked_pe = torch.stack(padded_pe, dim=0)

        masked_output = self.image_feature_embedding(masked_output)

        masked_output = masked_output + masked_pe

        masked_output = self.ln(masked_output)  # trying layernorm before first input into cross attention

        add_pos = True
        for i in range(len(self.cross_attns)):

            if i == len(self.cross_attns) - 1:
                add_pos = False

            if not self.box:
                masked_output, prompts = self.cross_attns[i](masked_output, prompts, original_prompts, masked_pe,
                                                             add_pos, padding_mask)
            else:
                masked_output, prompts = self.cross_attns[i](masked_output, prompts, original_prompts, masked_pe,
                                                             add_pos, point_mask)

        masked_output = self.embedding_to_feature(masked_output)

        images_input = images_input.masked_scatter(point_map, masked_output)

        images_input = images_input.permute(0, 3, 1, 2)

        return images_input, prompts