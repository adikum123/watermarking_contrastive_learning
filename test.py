import torch

msg = torch.randint(0, 2, (4, 1, 15), device=torch.device("cpu")).float() * 2 - 1
print(msg)
