import torch
from torch import nn
import einops
from UNet_model import encoder_block, conv_block, decoder_block, norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[2, 4, 8, 16], d_model=256, norm_name='batch',):
        super().__init__()

        self.promptSelfAttention = PromptSelfAttention(device,d_model,)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c, norm_name)
        self.e2 = encoder_block(base_c, base_c * kernels[0], norm_name)
        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1], norm_name)
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2], norm_name)

        """ Bottleneck """
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3], norm_name)

        self.b_to_embed = nn.Linear(96 * 16 * 16, d_model, bias=False)

        self.embed_to_b = nn.Linear(d_model, base_c * kernels[3] * 16 * 16, bias=False)

        self.crossattention = PromptImageCrossAttention(device,d_model,)
        """ Decoder """
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2], norm_name)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1], norm_name)
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0], norm_name)
        self.d4 = decoder_block(base_c * kernels[0], base_c, norm_name)

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)

    def forward(self, images, prompts):
        prompts = self.promptSelfAttention(prompts)
        """ Encoder """
        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """

        b = self.b(p4)

        b = einops.rearrange(b, 'b c h w  -> b (c h w)')

        b = self.b_to_embed(b)
        b = self.crossattention(b, prompts)

        b = self.embed_to_b(b)

        b = einops.rearrange(b, 'b (c h w) -> b c h w ', c=96, h=16, w=16)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)


        outputs = self.outputs(d4)

        return outputs


class Embeddings(nn.Module):

    def __init__(self,device, d_model, num_labels,):
        super().__init__()
        self.d_model = d_model

        self.num_labels = num_labels

        self.embedding = nn.Embedding(num_labels, d_model,dtype=torch.float32,device=device)

    def forward(self, x):
        return self.embedding(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.norm(x)


class FFN(nn.Module):

    def __init__(self, d_model: int, dropout=0.1, d_ff=None) -> None:
        super().__init__()
        if not d_ff:
            d_ff = 4 * d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model=256, num_labels=2, num_heads=4, d_position=2, dropout=0.1):
        super().__init__()

        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.attn_layer = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.res_connection = ResidualConnection(d_model, dropout=dropout)
        self.FFN = FFN(d_model, dropout=dropout)

    def forward(self, x):  # x_shape = B,L+1,d_model

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        attn_output = self.attn_layer(Q, K, V, need_weights=False)
        x = self.res_connection(x, attn_output)
        x = self.FFN(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.w_ki = nn.Linear(d_model, d_model, bias=False)
        self.w_qp = nn.Linear(d_model, d_model, bias=False)
        self.w_vp = nn.Linear(d_model, d_model, bias=False)
        self.w_kp = nn.Linear(d_model, d_model, bias=False)
        self.w_qi = nn.Linear(d_model, d_model, bias=False)
        self.w_vi = nn.Linear(d_model, d_model, bias=False)

        self.cross_attn_layer1 = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.res_connection = ResidualConnection(d_model, dropout=dropout)
        self.FFN = FFN(d_model, dropout=dropout)
        self.cross_attn_layer2 = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.res_connection2 = ResidualConnection(d_model, dropout=dropout)
        self.FFN2 = FFN(d_model, dropout=dropout)

    def forward(self, image, prompt):  # image shape = [B,1,d_model], prompt_shape [B,1,d_model

        Q_i = self.w_qi(image)
        K_p = self.w_kp(prompt)
        V_p = self.w_vp(prompt)
        prompt_attn_output = self.cross_attn_layer1(Q_i, K_p, V_p, need_weights=False)
        prompt = self.res_connection(prompt, prompt_attn_output)
        prompt = self.FFN(prompt)

        Q_p = self.w_qp(prompt)
        K_i = self.w_ki(image)
        V_i = self.w_vi(image)

        image_attn_output = self.cross_attn_layer2(Q_p, K_i, V_i, need_weights=False)
        image = self.res_connection2(image, image_attn_output)
        image = self.FFN2(image)

        return image, prompt


class PromptImageCrossAttention(nn.Module):

    def __init__(self,device, d_model, num_heads=4, num_layers=6, dropout=0.1,):
        super().__init__()
        assert d_model % num_heads == 0, "change params s.t. d_model is divisible by num_heads"

        self.device= device

        self.d_model = d_model
        self.num_layers = num_layers
        self.ImNorm = nn.LayerNorm(d_model)
        self.PNorm = nn.LayerNorm(d_model)
        self.ImagePromptEmbeddings = Embeddings(device,d_model, 2,)
        self.cross_attns = nn.ModuleList([CrossAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, image, prompt):  # image shape [B,d_model] prompt shape [B,d_model]
        B, d = image.shape
        image = image.unsqueeze(1)  # [B,1,d_model]
        prompt = prompt.unsqueeze(1)  # [B,1,d_model]
        image_label = self.ImagePromptEmbeddings(torch.zeros(B, dtype=torch.int).to(self.device)).unsqueeze(1)  # [B,1,d_model]
        prompt_label = self.ImagePromptEmbeddings(torch.ones(B, dtype=torch.int).to(self.device)).unsqueeze(1)  # [B,1,d_model]
        image += image_label
        prompt += prompt_label
        image = self.ImNorm(image)
        prompt = self.PNorm(prompt)

        for i in range(self.num_layers):
            image, prompt = self.cross_attns[i](image, prompt)

        return image.squeeze(1)


class PromptSelfAttention(nn.Module):

    def __init__(self,device, d_model=256, num_labels=3, num_heads=4, num_layers=4,d_position=2, d_embedding=254, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "change params s.t. d_model is divisible by num_heads"
        assert d_model - d_embedding == d_position, 'need to extra dimensions for x,y coords'
        self.device = device

        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.labelEmbeddings = Embeddings(device,d_embedding, num_labels,)
        self.empty_vector_embedding = Embeddings(device,d_model, 1,)
        self.self_attns = nn.ModuleList(
            [MultiHeadAttentionLayer(d_model, num_labels, num_heads, d_position, dropout) for _ in range(num_heads)])

    def forward(self, prompts):  # x should be a tensor of [B,C,L] of 'channels' (x,y,label)

        B, C, L = prompts.shape
        prompts = prompts.reshape(B * L, C)

        x_pos = prompts[:, 0].unsqueeze(1)  # [B*L,1]

        y_pos = prompts[:, 1].unsqueeze(1)  # [B*L,1]

        labels = prompts[:, 2]

        label_embeddings = self.labelEmbeddings(labels.to(torch.int))  # [B*L,d_model-d_position]

        embeddings_and_position = torch.cat((label_embeddings, x_pos, y_pos), 1)

        input = torch.reshape(embeddings_and_position, (B, L, self.d_model))

        empty_vector = self.empty_vector_embedding(torch.tensor([0]).to(self.device)).unsqueeze(0)  # shape [1,1,d_model]

        empty_vector = empty_vector.repeat((B, 1, 1))  # shape [B,1,d_model]
        input = torch.cat((empty_vector, input), dim=1)  # shape [B,L+1,d_model]

        input = self.norm(input)

        for i in range(self.num_layers):
            input = self.self_attns[i](input)

        return input[:, 0, :]


#
# model = PromptSelfAttention(num_heads=2,num_layers=2,d_model=16)
#
# input = torch.tensor([[(1,2,1),(1,1,0),(0,0,1)],[(1,2,1),(1,1,1),(0,0,1)]])
# input = torch.transpose(input,1,2)
# print(input.shape)
# print(input[0,:,1])
# print(model(input))

unet = PromptUNet(device,3, 1, kernels=[2, 4, 6, 8], base_c=12,).to(device)

input = torch.randn((2, 3, 512, 512)).to(device)
prompt_input = torch.tensor([[[1, 2, 1], [1, 1, 0], [0, 0, 1]], [[1, 2, 1], [1, 1, 1], [0, 0, 1]]]).to(device)
print(unet(input, prompt_input).shape)
