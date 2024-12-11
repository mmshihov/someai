import torch

### ### ###
# тестим косинусную дистанцию
### ### ###
x = torch.randn(4)
y = x.clone()
y[1] = 1000

print(x, y)

# пошли ебашки: dim=0? что?
cossim1 = torch.nn.CosineSimilarity(dim=-1)
cossim0 = torch.nn.CosineSimilarity(dim=0)

output = cossim1(x, y)
print(output)

output = cossim0(x, y)
print(output)

### ### ###
# тестим косинусную дистанцию
### ### ###

x = torch.tensor(
    [        
        [[1., 2., 3], [4., 5., 6.]], 
        [[7., 8., 9], [8., 7., 6.]], 
    ]
)

y = torch.tensor(
    [        
        [[7., 8., 9], [8., 7., 6.]], 
        [[1., 2., 3], [4., 5., 6.]], 
    ]
)

# гениально, что вектор, считается "по измерению", т.к. измерение -- это ось и все вектора вектора будут длины, равной размеру этого измерения
# и будет посчитан тензор, размерностью на одну менше, заполненный косинусными расстояниями т.е. если укажем dim=1
cossim = torch.nn.CosineSimilarity(dim=2)
output = cossim(x, y)
print(output)


x = torch.tensor([1., 1.])
y = torch.tensor([-1., -1.])

cossim = torch.nn.CosineSimilarity(dim=2)
output = cossim(x, y)
print(output) # -1 т.е. косинус в диапазоне -1 до 1
