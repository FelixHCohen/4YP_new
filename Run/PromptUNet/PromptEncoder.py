import torch
from torch import nn
from PromptUNet.PromptEmbedding import *
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
            y = self.dropout(sublayer(x))
            return self.norm(x+y)

class FFN(nn.Module):

    def __init__(self, d_model: int, dropout=0.1, d_ff=None) -> None:
        super().__init__()
        if not d_ff:
            d_ff = 4 * d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.m = torch.nn.ReLU()
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.m(self.linear_1(x))))


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()

        # self.w_k = nn.Linear(d_model, d_model, bias=False)
        # self.w_q = nn.Linear(d_model, d_model, bias=False)
        # self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.attn_layer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)



    def forward(self, x):  # x_shape = B,L+1,d_model

        # Q = self.w_q(x)
        # K = self.w_k(x)
        # V = self.w_v(x)
        # attn_output = self.attn_layer(Q, K, V, need_weights=False)
        attn_output = self.attn_layer(x,x,x, need_weights=False)
        return attn_output[0]

class SelfAttentionBlock(nn.Module):

    def __init__(self, d_model=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.MHattention = MultiHeadAttentionLayer(d_model,num_heads,dropout)
        self.FFN = FFN(d_model,dropout)
        self.res_connection1 = ResidualConnection(d_model,dropout)
        self.res_connection2 = ResidualConnection(d_model,dropout)

    def forward(self,x):
        x = self.res_connection1(x,self.MHattention)
        x = self.res_connection2(x,self.FFN)
        return x
class PromptEncoder(nn.Module):
     def __init__(self,device,pe_layer,d_model=384, input_image_size=(512,512), num_heads = 8,num_blocks=4, dropout=0.1,box=False):
        super().__init__()
        self.device = device
        if box:
            self.promptEmbedder = BoxPromptEmbedder(pe_layer,d_model,input_image_size,device)
        else:
            self.promptEmbedder = PromptEmbedder(pe_layer,d_model,input_image_size,device)

        self.self_attns = nn.ModuleList([SelfAttentionBlock(d_model, num_heads,dropout) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(d_model)
     def forward(self,points,labels): # point tensor should be BxLx(x,y) label tensor should be BxLx1
         embedded_points = self.promptEmbedder(points,labels,pad=False)
         embedded_points = self.ln(embedded_points)
         embedded_points_original = torch.clone(embedded_points)

         for i in range(len(self.self_attns)):
             embedded_points = self.self_attns[i](embedded_points)

         embedded_points_output = embedded_points + embedded_points_original
         return embedded_points_output,embedded_points_original

