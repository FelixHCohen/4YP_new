import torch
from torch import nn
from PromptUNet.PromptEncoder import *
class ResidualCrossConnection(ResidualConnection):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
            super().__init__(d_model,dropout)

    def forward(self,x,y,sublayer):
        output = self.dropout(sublayer(x,y))
        return self.norm(x + output)

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



    def forward(self, q_tensors,kv_tensors):  # x_shape = B,L+1,d_model

        Q = self.w_q(q_tensors)
        K = self.w_k(kv_tensors)
        V = self.w_v(kv_tensors)
        attn_output = self.attn_layer(Q, K, V, need_weights=False)

        return attn_output[0]


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

    def forward(self,images,prompts_input,original_prompts,pos_encodings,add_pos=True):

        prompts_input = self.res_connection1(prompts_input,images,self.CrossAttention1)
        prompts_input = self.res_connection2(prompts_input,self.FFN)
        images = self.res_connection3(images,prompts_input,self.CrossAttention2)
        images = self.res_connection4(images,pos_encodings,add_pos,self.FFN2)
        prompts_output = prompts_input + original_prompts
        return images,prompts_output

class ImageEncoder(nn.Module):

    def __init__(self,device,embeddings,d_model=128,d_image_in=576, num_heads=8,num_blocks=6, dropout=0.1,size=(32,32)):
        super().__init__()
        self.device = device
        self.d_model=d_model
        self.image_feature_embedding = nn.Linear(d_image_in,d_model)
        self.embedding_to_feature = nn.Linear(d_model,d_image_in)
        self.embeddings = embeddings
        self.cross_attns = nn.ModuleList([CrossAttentionBlock(d_model,num_heads,dropout) for _ in range(num_blocks)])
        self.size = size
        self.ln = nn.LayerNorm(d_model)



    def forward(self,images_input,prompts,original_prompts):
        B,D,H,W = images_input.shape
        grid = torch.ones((H, W), device=self.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / H
        x_embed = x_embed / W

        pe = self.embeddings._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        pe =  pe.permute(2, 0, 1)  # d_model x H x W

        pe = pe.unsqueeze(0).repeat(B,1,1,1) # shape Bxd_modelxHxW



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
        images = self.ln(images)  # trying layernorm before first input into cross attention
        add_pos = True
        for i in range(len(self.cross_attns)):
            if i == len(self.cross_attns)-1:
                add_pos = False
            images,prompts = self.cross_attns[i](images,prompts,original_prompts,pe,add_pos)

        images = self.embedding_to_feature(images)
        images = torch.reshape(images,(B,D,H,W))

        return images,prompts
