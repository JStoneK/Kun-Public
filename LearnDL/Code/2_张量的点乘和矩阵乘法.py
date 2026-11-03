"""
点乘：
    要求：两个张量的维度必须相同
    点乘是两个张量的对应元素相乘
    API：
        t1*t2
        t1.mul(t2)

矩阵乘法：
    要求：两个张量的维度必须是(a,m)和(m,p)->(a,p)
    线代
    API：
        t1@t2
        t1.matmul(t2)
        t1.dot(t2) :只针对于一维标量 点乘后求和
"""

import torch

def dm1():
    """
    点乘
    """
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    print(t1)

    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(t2)

    #t3 = t1*t2
    t3 = t1.mul(t2)
    print(t3)



def dm2():
    """
    矩阵乘法
    """
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    #print(t1)

    t2 = torch.tensor([[1,1],[1,1],[1,1]])
    #print(t2)

    #t3 = t1@t2
    #t3 = t1.matmul(t2)

    t4 = torch.tensor([1,2,3])
    t5 = torch.tensor([4,5,6])
    print(t4)
    print(t5)

    t6 = t4.dot(t5)
    print(t6)


if __name__ =="__main__":
    #dm1()
    dm2()