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
            attn_output = self.attn_layer(Q,K,V,need_weights=False,key_padding_mask=attn_mask)
        else:
            attn_output = self.attn_layer(Q, K, V, need_weights=False)

        return attn_output[0]

class CrossAttentionBlockBatch(nn.Module):
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
        prompts_input = self.res_connection1(prompts_input, images, self.CrossAttention1)
        prompts_input = self.res_connection2(prompts_input, self.FFN)
        images = self.res_connection3(images, prompts_input, self.CrossAttention2)
        images = self.res_connection4(images, pos_encodings, add_pos, self.FFN2)
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

        prompts_input = self.res_connection1(prompts_input,images,self.CrossAttention1,attn_mask) # if using masked encoder need to hide empty image K-V pairs
        prompts_input = self.res_connection2(prompts_input,self.FFN)
        images = self.res_connection3(images,prompts_input,self.CrossAttention2)
        images = self.res_connection4(images,pos_encodings,add_pos,self.FFN2)
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



class maskedImageEncoder(ImageEncoder):
    def __init__(self, device, embeddings, d_model=128, d_image_in=576, num_heads=8, num_blocks=6, dropout=0.1,radius=10):
        super().__init__(device,embeddings,d_model,d_image_in,num_heads,num_blocks,dropout)
        self.num_heads = num_heads
        self.radius = radius

    def add_additional(self,point_map,H,W,b):
        additional_list = []
        for i in range(H):
            for j in range(W):
                if point_map[b,i,j] == 0:
                    for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                        if i + di >= 0 and i + di < H and j + dj >= 0 and j + dj < W:
                            if point_map[b][i + di][j + dj] == 1:
                                additional_list.append([i, j])
                                break
        return additional_list

    def forward(self, images_input, prompts, original_prompts, points,):
        B, D, H, W = images_input.shape

        pe = self.pos_emb_grid(B,H,W)


        points /= (512/H) # assuming input image is square

        num_points = points.shape[1]
        L = num_points*(self.radius+1)**2
        masked_output = torch.zeros((B,L,D)).to(self.device)
        masked_pe = torch.zeros((B,L,self.d_model)).to(self.device)
        images_input = torch.permute(images_input,(0, 2, 3, 1))  # rearange to BxHxWxD so when we flatten vectors are sorted by batch
        #images_input = self.image_feature_embedding(images_input)

        point_map = torch.zeros((B,H, W))

        for b in range(B):
            count = 0
            for point_idx in range(num_points):
                i,j = points[b,point_idx,0].to(torch.int),points[b,point_idx,1].to(torch.int)

                i_min = max(0,i- self.radius//2)
                j_min = max(0,j- self.radius//2)
                i_max = min(H,i+ self.radius//2)
                j_max = min(W,j+ self.radius//2)

                point_map[b,i_min:i_max+1,j_min:j_max+1].fill_(1)



            count = int(torch.sum(point_map[b,:,:]).item())

            while count < L:
                additional_list = self.add_additional(point_map,H,W,b)

                len_additional = len(additional_list)

                if len_additional > L - count:
                    to_delete = set(random.sample(range(len(additional_list)), len_additional - L + count))
                    additional_list = [x for i, x in enumerate(additional_list) if not i in to_delete]
                for i,j in additional_list:
                    point_map[b][i][j] = 1

                count += len_additional # if former condition satisfied then count = L therefore count > L satisfies same cond


        point_map = point_map.bool().to(self.device).unsqueeze(3)
        pe = pe.permute(0,2,3,1)


        masked_output = torch.reshape(torch.masked_select(images_input,point_map),(B,L,D))
        masked_pe = torch.reshape(torch.masked_select(pe,point_map),(B,L,self.d_model))


        masked_output = self.image_feature_embedding(masked_output)


        masked_output = masked_output + masked_pe

        masked_output = self.ln(masked_output)  # trying layernorm before first input into cross attention

        add_pos = True
        for i in range(len(self.cross_attns)):
            if i == len(self.cross_attns) - 1:
                add_pos = False
            masked_output, prompts = self.cross_attns[i](masked_output, prompts, original_prompts, masked_pe, add_pos)


        masked_output = self.embedding_to_feature(masked_output)

        images_input.masked_scatter(point_map,masked_output)


        images_input = images_input.permute(0,3,1,2)

        return images_input,prompts
