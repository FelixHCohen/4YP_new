import torch
from torch import nn
from PromptUNet.ImageEncoder import *
from PromptUNet.PromptEncoder import *
import math

def norm(input: torch.tensor, norm_name: str):
    if norm_name == 'layer':
        normaliza = nn.LayerNorm(list(input.shape)[1:])
    elif norm_name == 'batch':
        normaliza = nn.BatchNorm2d(list(input.shape)[1])
    elif norm_name == 'instance':
        normaliza = nn.InstanceNorm2d(list(input.shape)[1])

    normaliza = normaliza.to(f'cuda:0')

    output = normaliza(input)

    return output
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, norm_name):
        super().__init__()

        self.norm_name = norm_name

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        # self.norm1 = norm

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        # self.norm2 = norm

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        # x = self.norm1(x, self.norm_name)
        x = norm(x, self.norm_name)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.norm2(x, self.norm_name)
        x = norm(x, self.norm_name)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, norm_name):
        super().__init__()

        self.conv = conv_block(in_c, out_c, norm_name)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, norm_name):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, norm_name)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class PromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[6, 12, 24,48,], d_model=64,num_prompt_heads=4,num_prompt_blocks=2,num_heads=8,num_blocks=6,img_size=(512,512),dropout=0.1, norm_name='batch'):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,3,dropout)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c, norm_name)
        self.e2 = encoder_block(base_c, base_c * kernels[0], norm_name)
        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1], norm_name)
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2], norm_name)
        #self.e5 = encoder_block(base_c*kernels[2],base_c*kernels[3],norm_name)
        """ Bottleneck """
        #self.b1 = conv_block(base_c * kernels[3], base_c * kernels[4], norm_name)
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3], norm_name)

        x,y = img_size
        bottleneck_size = (x//2**(len(kernels)),y//2**(len(kernels)))
        L = bottleneck_size[0]*bottleneck_size[1]
        size_2 = (64,64) #note to self size not needed anymore
        #self.promptImageCrossAttention = ImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in,num_heads,2,dropout,bottleneck_size)
        self.promptImageCrossAttention_post = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in//2, num_heads, 8,dropout, size_2)
        self.promptImageCrossAttention_post_post = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,num_heads, 4, dropout, size_2)

        """ Decoder """
       # self.dnew = decoder_block(base_c * kernels[4], base_c * kernels[3], norm_name)
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2], norm_name)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1], norm_name)
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0], norm_name)
        self.d4 = decoder_block(base_c * kernels[0], base_c, norm_name)

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

       # if train_attention:
        #    p4,prompts = self.promptImageCrossAttention_pre(p4,prompts,original_prompts)
        #s5,p5 = self.e5(p4)
        """ Bottleneck """

        #b = self.b1(p5)
        #print(f'b shape: {b.shape}')

        b = self.b(p4)

       # if train_attention:
        #    b,prompts = self.promptImageCrossAttention(b,prompts,original_prompts)


        #print(f'prompts2\n\n\n{prompts2}')
        """ Decoder """



        d1 = self.d1(b, s4)

        if train_attention:
            d1, prompts = self.promptImageCrossAttention_post(d1, prompts,original_prompts)



        #print(f'prompts3:\n\n\n{prompts3}')
        d2 = self.d2(d1, s3)

        if train_attention:
            d2,prompts = self.promptImageCrossAttention_post_post(d2,prompts,original_prompts)
      #  if train_attention:
       #     d2,prompts_add_3 = self.promptImageCrossAttention3(d2,prompts3)
        d3 = self.d3(d2, s2)
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
        print(f'point loss: {self.beta*self.alpha*loss1} gen loss: {self.beta*(1-self.alpha)*loss2}')
        return self.beta*(self.alpha*loss1 + (1-self.alpha)*loss2)





