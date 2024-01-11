import torch

mask = torch.tensor([[[True,False],[False,False],[False,True]],
                     [[False,False],[True,False],[False,True]],
                     [[False,True],[False,False],[False,True]],
                     [[False,False],[True,False],[False,True]],
                     [[False,False],[False,True],[False,True]],
                     [[False,False],[False,False],[True,True]],
                     [[False,False],[True,False],[False,True]],
                     [[False,False],[True,False],[False,True]]])
mask = mask.unsqueeze(3)
print(mask.shape)
input = torch.tensor([[[[1,13,1,1],[2,14,1,1]],[[3,15,1,1],[4,16,1,1]],[[5,17,1,1],[6,18,1,1]]],[[[7,19,1,1],[8,20,1,1]],[[9,21,1,1],[10,22,1,1]],[[11,23,1,1],[12,24,1,1]]]])
input = input.repeat(4,1,1,1)
print(input.shape)
masked_input = torch.masked_select(input,mask)
masked_input = masked_input.reshape((8,2,4))
print(mask[2,0,1,0])
print(f'selected: {masked_input[2,:,:]}')
print(f' og1: {input[2,0,1,:]}')
print(f' og2: {input[2,2,1,:]}')
output = torch.zeros((8,2,4)).long()
#output[2,:,:] = masked_input[2,:,:]
final_output = input.masked_scatter(mask,output)
print(final_output[2,0,1,:])
print(final_output[2,2,1,:])
