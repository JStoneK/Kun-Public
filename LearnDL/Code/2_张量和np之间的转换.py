"""
涉及到的情景：
    1. 张量转为np nd数组对象
        张量对象.numpy() 共享内存
        张量对象.numpy().copy() 不共享内存
    2. np nd数组对象转为张量
        from_numpy(数组对象，类型) 共享内存
        torch.tensor(数组对象，类型) 不共享内存
    3.从标量张量中提取内容
        标量张量.item() 
掌握：
    张量转np : 张量对象.numpy() 
    np转张量 : torch.tensor(数组对象，类型)
    从标量张量(1维张量)中提取内容 : 标量张量.item()
"""

import torch 
import numpy as np

def dm1():
    t1 = torch.tensor([1,2,3,4,5])
    # 共享内存 t2 = t1.numpy()
    t2 = t1.numpy().copy() #不共享内存
    print(f"t1 = {t1} , type  = {type(t1)}")
    print(f"t2 = {t2} , type  = {type(t2)}")
    # test
    t2[0] = 100
    print(f"t1 = {t1} , type  = {type(t1)}")
    print(f"t2 = {t2} , type  = {type(t2)}")


def dm2():
    n1 = np.array([11,22,33])
    print(f"n1 = {n1} , type  = {type(n1)}")

    t1 = torch.from_numpy(n1) # 共享内存
    print(f"t1 = {t1} , type  = {type(t1)}")

    t2 = torch.tensor(n1) # 不共享内存
    print(f"t2 = {t2} , type  = {type(t2)}")

    n1[0] = 100
    print(f"n1 = {n1} , type  = {type(n1)}")
    print(f"t1 = {t1} , type  = {type(t1)}")
    print(f"t2 = {t2} , type  = {type(t2)}")
def dm3():
    t1 = torch.tensor(100.1)
    print(f"t1 = {t1} , type  = {type(t1)}")

    a = t1.item()
    print(f"a = {a} , type  = {type(a)}")
    
if __name__ == "__main__":
    #dm1()
    #dm2()
    dm3()