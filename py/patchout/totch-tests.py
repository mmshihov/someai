import torch

### ### ###
# тестим косинусную дистанцию
x = torch.randn(4)
y = torch.randn(4)

print(x, y)

cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output = cossim(x, y)

print(x, y)
