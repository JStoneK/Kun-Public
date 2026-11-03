"""
API:
    sum(),max(),min(),mean() 都有dim参数，dim表示在哪个维度上进行运算，0表示按照列，1表示按照行
    pow(),sqrt(),exp(),log(),log2(),log10() ,都没有dim参数，对每个元素进行运算
掌握：
    sum(),max(),min(),mean(),pow()
"""


import torch 
t1 = torch.tensor(
    [[1,2,3],
     [4,5,6]]
     ,dtype=torch.float32
)

#print(t1.sum(dim = 0))
#print(t1.sum(dim = 1))
#print(t1.sum())

#print(t1.max(dim = 0))
#print(t1.max(dim = 1))
#print(t1.max())


#print(t1.mean(dim = 0))
#print(t1.mean(dim = 1))
#print(t1.mean())

print(t1.pow(2))
print(t1**2)


print(t1.sqrt())

print(t1.exp())

print(t1.log())
print(t1.log2())
print(t1.log10())
