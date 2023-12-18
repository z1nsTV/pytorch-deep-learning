import torch
import numpy as np;
import time;

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
float_16_torch = torch.tensor([1,3,6], dtype=torch.float16)
long_torch = torch.tensor([1,3,6], dtype=torch.long)

#Matrix multiplication (dot product) a.b
#element-wise, matmul (dot product) @

first = torch.tensor([1,2,3])
second = torch.tensor([7,9,11])
# print(first*second) # element-wise
# print(torch.matmul(first, second)) # dot product
big = torch.tensor([[158,568,655],
                    [12,5668,744],
                    [3,32,45142]])
now = time.time()
val = 0
for i in range(len(first)):
    val += first[i] * first[i]
# print(f"Calculation of val: {val} by hand took {(time.time() - now)*1000:.12f} ms")

now = time.time()
mul = 0
mul = torch.matmul(first, first);
# print(f"Calculation of val: {val} by matmul took {(time.time() - now)*1000:.12f} ms")

now = time.time()
mul = 0
mul = torch.matmul(big, big);
# print(f"Calculation of val: {val} by matmul took {(time.time() - now)*1000:.12f} ms")

#1. Inner dimension must match 
#(3,2) @ (2,3) will work -> inner 2) and (2
#(3,2) @ (3,2) won't work -> inner 2) and (3
# not_working_torch = torch.matmul(torch.rand(2,3), torch.rand(2,3))
# print(not_working_torch)
#RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)

#2. Resulting matrix has shape of outer dimensions
outer_dimension = torch.matmul(torch.rand(4,3), torch.rand(3,11))
print(outer_dimension.shape)
