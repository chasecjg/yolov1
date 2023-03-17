import torch
data = torch.rand(3,7,7,30)
coo_mask = data[:, :, :, 4] > 0
print(coo_mask.shape)