from PromptUNet.PromptUNet import *

class MaskedPromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[6, 12, 24,48,],attention_kernels=[4,2,4,4], d_model=64,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1,batch_norm=False,box=False):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device

        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout,box=box)
        """ Encoder """

        # self.e1 = encoder_block(in_c, base_c,batch_norm)
        # self.e2 = encoder_block(base_c, base_c * kernels[0],batch_norm)
        # self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1],batch_norm)

        self.e1 = encoder_block(in_c, base_c, batch_norm)
        self.e2 = conv_block(base_c,base_c*kernels[0],batch_norm)
        self.e2_p = nn.MaxPool2d((2, 2))
        self.e3 = conv_block(base_c*kernels[0],base_c*kernels[1],batch_norm)
        self.e3_p =  nn.MaxPool2d((2, 2))


        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2],batch_norm)
        #self.e5 = encoder_block(base_c*kernels[2],base_c*kernels[3])
        """ Bottleneck """
        #self.b1 = conv_block(base_c * kernels[3], base_c * kernels[4])
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],batch_norm)

        self.masked_encoder_attention_1 = maskedImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in//8,num_heads,2,dropout,radius=32,)
        self.encoder_attention_2 = ImageEncoder(self.device, self.pe_layer, d_model = self.d_model, d_image_in = self.d_image_in//4,num_heads = num_heads, num_blocks = attention_kernels[3],dropout = dropout,batch=batch_norm)
        self.encoder_attention_3 = ImageEncoder(self.device, self.pe_layer, d_model = self.d_model, d_image_in = self.d_image_in//2,num_heads = num_heads, num_blocks = attention_kernels[2],dropout = dropout,batch=batch_norm)


        self.promptImageCrossAttention_1 = ImageEncoder(self.device, self.pe_layer, d_model = self.d_model, d_image_in = self.d_image_in//2,num_heads = num_heads, num_blocks = attention_kernels[2],dropout = dropout,batch=batch_norm)
        self.promptImageCrossAttention_2_mask = maskedImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,num_heads,attention_kernels[3], dropout,radius=64)
        self.crossattention_final_mask = maskedImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in//8,num_heads,2,dropout,radius=32,)
        """ Decoder """

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

        if train_attention:
            s2,
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """

        b = self.b(p4)


        """ Decoder """

        d1 = self.d1(b, s4)

        if train_attention:
            d1, prompts = self.promptImageCrossAttention_1(d1,prompts,original_prompts)


        d2 = self.d2(d1, s3)

        if train_attention:
            d2,prompts = self.promptImageCrossAttention_2_mask(d2,prompts,original_prompts,points)


        d3 = self.d3(d2, s2)

        if train_attention:
            d3,prompts = self.crossattention_final_mask(d3,prompts,original_prompts,points)

        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs



