"""
张量的基本创建：
    torch.tensor() 根据标量创建张量
    torch.Tensor() 根据标量创建张量，与torch.tensor()不同的是，torch.Tensor()返回的张量是不可训练的张量     可以指定形状
    torch.IntTensor() 根据标量创建整数张量，torch.FloatTensor() 根据标量创建浮点数张量，torch.BoolTensor() 根据标量创建布尔值张量
需要掌握的：
    tensor(值，类型)
"""


import torch
import numpy as np

def dm1():
    
    # 根据标量创建张量
    t1 = torch.tensor(10)
    print(f"t1 = {t1} , type  = {type(t1)}")
    
    # 根据列表创建张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f"t2 = {t2} , type  = {type(t2)}")
    
    # 根据numpy数组创建张量
    data = np.random.randint(0,10,size=(2,3))
    t3 = torch.tensor(data,dtype = torch.float) 
    print(f"t3 = {t3}, type = {type(t3)}")

def dm2():
    
    # 根据标量创建张量
    t1 = torch.Tensor(10)
    print(f"t1 = {t1} , type  = {type(t1)}")
    
    # 根据列表创建张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f"t2 = {t2} , type  = {type(t2)}")
    
    # 根据numpy数组创建张量
    data = np.random.randint(0,10,size=(2,3))
    t3 = torch.Tensor(data) 
    print(f"t3 = {t3}, type = {type(t3)}")

    #直接创建张量
    t4 = torch.Tensor(2,3)
    print(f"t4 = {t4}, type = {type(t4)}")
    
def dm3():
    # 根据标量创建张量
    t1 = torch.IntTensor(10)
    print(f"t1 = {t1} , type  = {type(t1)}")
    
    # 根据列表创建张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.IntTensor(data)
    print(f"t2 = {t2} , type  = {type(t2)}")
    
    # 根据numpy数组创建张量
    data = np.random.randint(0,10,size=(2,3))
    t3 = torch.IntTensor(data) 
    print(f"t3 = {t3}, type = {type(t3)}")

    #类型不匹配尝试自动转换类型
    data = np.random.randint(0,10,size=(2,3))
    t4 = torch.FloatTensor(data) 
    print(f"t4 = {t4}, type = {type(t4)}")



if __name__ == "__main__":
    #dm1()
    #dm2()
    dm3()