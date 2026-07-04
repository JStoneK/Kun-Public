"""
创建指定类型的张量

涉及的函数：
    type(torch支持的类型)
    half()/double()/float()/short()/int()/long()

掌握的函数：
    type()
"""



import torch

#直接创建指定类型的张量
t1 = torch.tensor([1,2,3,4,5],dtype=torch.float32)
print(f"t1 = {t1}, 元素type = {t1.dtype}, 张量type = {type(t1)}") 


#创建好张量后做类型转换
#1.type()
t2 = t1.type(torch.int16)
print(f"t2 = {t2}, 元素type = {t2.dtype}, 张量type = {type(t2)}") 

#2.half()/double()/float()/short()/int()/long()

print(t2.half())
print(t2.double())
print(t2.float())
print(t2.short())
print(t2.int())
print(t2.long())
