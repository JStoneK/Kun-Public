import torch

a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
   
], dtype=torch.float32)

print(a.mean(dim=1))