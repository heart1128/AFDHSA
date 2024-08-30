import torch

input = torch.Tensor(16, 2048)

liner = torch.nn.Linear(2048, 1)

x = liner(input)

print(x.shape)