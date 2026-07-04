"""
创建全0/全1/指定值的张量

涉及到的函数：
 torch.ones() torch.ones_like() 创建全1张量
 torch.zeros() torch.zeros_like() 创建全0张量
 torch.full() torch.full_like() 创建指定值的张量
掌握的函数：
    zeros(),full()
"""

import torch




#创建全1张量
t1 = torch.ones(2,3)
print(f"t1 = {t1}, type = {type(t1)}")

t2 = torch.tensor([[1,2],[3,4],[5,6]])
print(f"t2 = {t2}, type = {type(t2)}")

t3 = torch.ones_like(t2)
print(f"t3 = {t3}, type = {type(t3)}")

#创建全0张量
t1 = torch.zeros(2,3)
print(f"t1 = {t1}, type = {type(t1)}")

t2 = torch.tensor([[1,2],[3,4],[5,6]])
print(f"t2 = {t2}, type = {type(t2)}")

t3 = torch.zeros_like(t2)
print(f"t3 = {t3}, type = {type(t3)}")

#创建指定值的张量

t1 = torch.full((2,3),100)
print(f"t1 = {t1}, type = {type(t1)}")

t2 = torch.tensor([[1,2],[3,4],[5,6]])
print(f"t2 = {t2}, type = {type(t2)}")

t3 = torch.full_like(t2,100)
print(f"t3 = {t3}, type = {type(t3)}")
