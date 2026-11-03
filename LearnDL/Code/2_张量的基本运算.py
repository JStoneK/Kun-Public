"""
涉及到的API
    add(),sub(),mul(),div(),neg() 加减乘除取反,取反就是变为负号
    add_(),sub_(),mul_(),div_() 原地加减乘除 可以修改数据，类似于pandas的inplace=True
掌握：
    可以用+-*/替换上面的API
    如果是张量和数值运算，则该数值会广播到张量的所有元素，也就是依次对每个元素进行运算
"""

import torch 



t1 = torch.tensor([1,2,3])
#t2 = t1.add(10)
#t2 = t1 + 10
#t1 += 10
#print(t1)
#t1 = t1 + 10
#print(t1)
#print(t2)
 
#t2 = t1.sub(1)
#t2 = t1.mul(2)
#t2 = t1.div(2)
#t2 = t1.neg()

t2 = t1 / 2
t3 = t1 // 2


print(t2)
print(t3)

