import torch
from torch import nn
from PromptUNet.ImageEncoder import *
from PromptUNet.PromptEncoder import *
import math

class conv_block(nn.Module):
    def __init__(self, in_c, out_c,batch_norm=False):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(out_c,affine=batch_norm)

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)


        self.norm2 = nn.BatchNorm2d(out_c,affine=batch_norm)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)


        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)

        x = self.norm1(x)


        x = self.relu(x)

        x = self.conv2(x)

        x = self.norm2(x)


        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c,batch_norm=False):
        super().__init__()

        self.conv = conv_block(in_c, out_c,batch_norm)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c,batch_norm=False):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c,batch_norm)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class PromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[6, 12, 24,48,],attention_kernels=[3,2,4,4], d_model=384,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1,batch_norm=False,box=False):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout,box)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c,batch_norm)
        self.e2 = encoder_block(base_c, base_c * kernels[0],batch_norm)
        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1],batch_norm)
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2],batch_norm)
        #self.e5 = encoder_block(base_c*kernels[2],base_c*kernels[3])
        """ Bottleneck """
        #self.b1 = conv_block(base_c * kernels[3], base_c * kernels[4])
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],batch_norm)


        self.promptImageCrossAttention_post = ImageEncoder(self.device, self.pe_layer, d_model = self.d_model, d_image_in = self.d_image_in//2,num_heads = num_heads, num_blocks = attention_kernels[2],dropout = dropout,batch=batch_norm)
        self.promptImageCrossAttention_post_post = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,num_heads,attention_kernels[3], dropout,batch=batch_norm)
        #self.crossattention_final = ImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in//8,num_heads,2,dropout,)
        """ Decoder """
       # self.dnew = decoder_block(base_c * kernels[4], base_c * kernels[3])
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2],batch_norm)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1],batch_norm)
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0],batch_norm)
        self.d4 = decoder_block(base_c * kernels[0], base_c,batch_norm)

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)




    def forward(self, images,points,labels,train_attention = True):
        #print(f'inputs\nimages:{images.device}\npoints{points.device}\nlabels:{labels.device}')
        if train_attention:
            prompts, original_prompts = self.promptSelfAttention(points, labels)

        """ Encoder """
        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """

        b = self.b(p4)


        """ Decoder """

        d1 = self.d1(b, s4)

        if train_attention:
            d1, prompts = self.promptImageCrossAttention_post(d1, prompts,original_prompts)


        d2 = self.d2(d1, s3)

        if train_attention:
            d2,prompts = self.promptImageCrossAttention_post_post(d2,prompts,original_prompts,)


        d3 = self.d3(d2, s2)

        #if train_attention:
        #    d3,prompts = self.crossattention_final(d3,prompts,original_prompts,points)

        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs


class SymmetricPromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[6, 12, 24,48,],attention_kernels=[3,2,4,4], d_model=384,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1,batch_norm=False,box=False):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout,box)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c,batch_norm)
        self.e2 = encoder_block(base_c, base_c * kernels[0],batch_norm)
        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1],batch_norm)
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2],batch_norm)
        #self.e5 = encoder_block(base_c*kernels[2],base_c*kernels[3])
        """ Bottleneck """
        #self.b1 = conv_block(base_c * kernels[3], base_c * kernels[4])
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],batch_norm)


        self.promptImageCrossAttention_post = ImageEncoder(self.device, self.pe_layer, d_model = self.d_model, d_image_in = self.d_image_in//2,num_heads = num_heads, num_blocks = attention_kernels[2],dropout = dropout,batch=batch_norm)
        self.promptImageCrossAttention_post_post = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,num_heads,attention_kernels[3], dropout,batch=batch_norm)
        #self.crossattention_final = ImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in//8,num_heads,2,dropout,)
        """ Decoder """
       # self.dnew = decoder_block(base_c * kernels[4], base_c * kernels[3])
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2],batch_norm)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1],batch_norm)
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0],batch_norm)
        self.d4 = decoder_block(base_c * kernels[0], base_c,batch_norm)

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)




    def forward(self, images,points,labels,train_attention = True):
        #print(f'inputs\nimages:{images.device}\npoints{points.device}\nlabels:{labels.device}')
        if train_attention:
            prompts, original_prompts = self.promptSelfAttention(points, labels)

        """ Encoder """
        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """

        b = self.b(p4)


        """ Decoder """

        d1 = self.d1(b, s4)

        if train_attention:
            d1, prompts = self.promptImageCrossAttention_post(d1, prompts,original_prompts)


        d2 = self.d2(d1, s3)

        if train_attention:
            d2,prompts = self.promptImageCrossAttention_post_post(d2,prompts,original_prompts,)


        d3 = self.d3(d2, s2)

        #if train_attention:
        #    d3,prompts = self.crossattention_final(d3,prompts,original_prompts,points)

        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs
class pointLoss(nn.Module):

    def __init__(self,radius):
        super().__init__()
        self.radius = radius
        self.sigma = radius//3
        self.init_loss = nn.CrossEntropyLoss(reduction='none')


    def gaussian(self,i, j, i_p, j_p,):

        # i1, j1: the coordinates of the pixel
        # i, j: the coordinates of the center
        # sigma: the standard deviation of the Gaussian
        # returns the value of the Gaussian function at pixel i1,j1
        d = torch.sqrt((i - i_p) ** 2 + (j - j_p) ** 2)  # distance between pixel and center
        f = torch.exp(-0.5 * (d / self.sigma) ** 2)   # Gaussian function reweighted s.t. it is 1 at d=0
        return f

    def forward(self,y_pred,y_true,points,point_labels,device):

        B,L,_ = points.shape
        _,_,H,W = y_true.shape
        y_pred = y_pred.softmax(dim=1)
        mask = torch.zeros((B,H,W)).to(device)
        y_true_one_hot = torch.nn.functional.one_hot(y_true.long(),num_classes=3).squeeze()
        #shape is BxHxWxC now
        y_true_one_hot = torch.permute(y_true_one_hot,(0,3,1,2))
        for b in range(B):
            for l in range(L):
                i_p,j_p = points[b,l,:]
                i_min = max(0,i_p-self.radius//2)
                i_max = min(H,i_p + 1 + self.radius//2)

                j_min = max(0,j_p-self.radius//2)
                j_max = min(W,j_p + self.radius//2)

                i_grid = list(range(int(i_min),int(i_max)))
                j_grid = list(range(int(j_min),int(j_max)))

                for i in i_grid:
                    for j in j_grid:

                        if y_true[b,0,i,j] == point_labels[b,l,0]:
                            mask[b,i,j] += self.gaussian(i,j,i_p,j_p)

        loss = self.init_loss(y_true_one_hot.to(torch.float32),y_pred)


        loss*=mask

        return torch.mean(loss)


class NormalisedFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(NormalisedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=3).squeeze()
        # # shape is BxHxWxC now
        # target_one_hot = torch.permute(target_one_hot, (0, 3, 1, 2))
        # print(f'target one hot shape: {target_one_hot.shape}')
        target = target.to(dtype=torch.int64)
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target_one_hot = target.view(-1, 1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target_one_hot)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt

        normalising_constant = torch.sum((1-pt)**self.gamma)

        loss /= normalising_constant

        if self.size_average: return loss.mean()
        else: return loss.sum()

class combine_loss(nn.Module):

    def __init__(self,loss1,loss2,alpha):
        super().__init__()
        self.l1 = loss1
        self.l2 = loss2
        self.alpha = alpha

    def forward(self,y_pred,y_true):
        return self.alpha*(self.l1(y_pred,y_true)) + (1-self.alpha)*(self.l2(y_pred,y_true))
class combine_point_loss(nn.Module):

    def __init__(self,pointloss,loss,alpha,beta):
        super().__init__()

        self.pointloss = pointloss
        self.loss = loss
        self.alpha = alpha
        self.beta = beta

    def forward(self,y_pred,y_true,points,labels,device):

        loss1 = self.pointloss(y_pred,y_true,points,labels,device)
        loss2 = self.loss(y_pred,y_true)
        pl = self.alpha*loss1
        gl = self.beta*loss2

        return pl+gl,pl,gl





