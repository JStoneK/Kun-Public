"""
创建线性和随机张量

涉及的函数：
 torch.arange() torch.linspace() 创建先行张量
 torch.random.initial_seed() torch.random.manual_seed()初始化随机种子
 torch.rand() torch.randn() 创建随机浮点张量
 torch.randint(low,high,size=()) 创建随机整数张量

掌握的函数:
    arange(),linspace(),manual_seed(),randint()
"""


import torch

def dm1():
    """
        创建线性张量
    """
    #指定范围 起始值 结束值（不包含结束值） 步长
    t1 = torch.arange(0,10,2)
    print(f"t1 = {t1}, type = {type(t1)}")

    #等差数列 起始值 结束值（包含结束值） 点数
    t2 = torch.linspace(0,10,6)
    print(f"t2 = {t2}, type = {type(t2)}")


def dm2():
    """
        创建随机张量
    """
    
    #初始化随机数种子
    #torch.initial_seed() #默认采用当前时间戳
    torch.random.manual_seed(100) #手动设置随机种子


    #创建随机张量
    #均匀分布0-1（不包含1）
    t1 = torch.rand(2,3)
    print(f"t1 = {t1}, type = {type(t1)}") 

    #正态分布随机张量
    t2 = torch.randn(2,3)
    print(f"t2 = {t2}, type = {type(t2)}") 


    #随机整数张量
    t3 = torch.randint(0,10,(2,3))
    print(f"t3 = {t3}, type = {type(t3)}")  

    
if __name__ == "__main__":
    #dm1()
    dm2()