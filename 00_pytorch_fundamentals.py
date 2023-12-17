import torch
import numpy as np;
#scalar
scalar = torch.tensor(7)
#vector 
vector = torch.tensor([7,7])
#MATRIX
MATRIX = torch.tensor([[7,8],
                       [5,4]])
#TENSOR
TENSOR = torch.tensor([[1,2,3],
                        [3,4,5],
                        [4,5,6]])
#ndim has one dimension (dimesions number of outside [] brackets pairings)
RANDOM_TENSOR = torch.rand(3,4)

random_image_tensor = torch.rand(size=(224,224,3)) #height, width, color channels (RGB)
#torch.dtype = torch.float32 default type 
zeros = torch.zeros(1,2,3)
#one dimension torch using arange
one_to_ten = torch.arange(1,11)
one_to_milion = torch.arange(start=0,end=1000000,step=10000)
#tensor-like
ten_zeros = torch.zeros_like(one_to_ten)
#RuntimeError: "check_uniform_bounds" not implemented for 'Long' when not dtype specified for torch.float32
ten_rand = torch.rand_like(one_to_ten, dtype=torch.float32)

#Data types

print(ten_rand)
