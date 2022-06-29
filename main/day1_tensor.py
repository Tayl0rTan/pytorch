import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data,dtype=torch.float)

shape = (2,3)

rand_tensor = torch.rand(shape,dtype=torch.float)

tensor = torch.rand(3,4)

tensor = torch.ones(4, 4)
tensor[:,1] = 0

t1 = torch.cat([tensor, tensor, tensor], dim=-2)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.mean()
agg_item = agg.item()

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")