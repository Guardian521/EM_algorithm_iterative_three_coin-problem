
import random
import numpy as np
import time
def coin_experiment(pi, p, q, n,seed=None):
    """
    模拟掷硬币实验
    pi: 硬币 A 正面出现的概率
    p: 硬币 B 正面出现的概率
    q: 硬币 C 正面出现的概率
    n: 实验次数
    """
    if seed is not None:
        random.seed(seed)
    results = []
    results_A = []
    results_B = []
    results_C = []
    for i in range(n):
        # 先掷硬币 A
        if random.random() < pi:
            # 选硬币 B
            coin = 'B'
            p_head = p
        else:
            # 选硬币 C
            coin = 'C'
            p_head = q

        # 接着掷选出的硬币
        if random.random() < p_head:
            results.append(1)
        else:
            results.append(0)

        # 记录每个硬币的正反面
        if coin == 'B':
            if random.random() < p:
                results_B.append(1)
            else:
                results_B.append(0)
            results_A.append(1)
        else:
            if random.random() < q:
                results_C.append(1)
            else:
                results_C.append(0)
            results_A.append(0)
    print(results_A,results_B,results_C)
    # 计算 A、B、C 硬币的正面概率
    p_A = sum(results_A) / len((results_A))
    p_B = sum(results_B) / len(results_B)
    p_C = sum(results_C) / len(results_C)

    return results, p_A, p_B, p_C


pi=0.7
p=0.3
q=0.6
n = 100
s=time.time()
print(f'开始模拟，模拟参数为pi={pi},p={p},q={q}……')
Y,pi,p,q=coin_experiment(pi, p, q, n)
Y= np.array(Y)
print(f'模拟结束，共模拟{n}次，耗时{time.time()-s}……')
print(f"模拟结果，pi={pi},p={p},q={q},整体实验Y=1概率={sum(Y)/len(Y)}")

#EM参数反推，任意设定初始参数
pi_0 = 0.9
p_0 = 0.9
q_0 = 0.9
epsiodes=100000 #迭代次数
count=1
while count<=epsiodes:
    mu = pi_0 * p_0**Y * (1-p_0)**(1-Y) / (pi_0 * p_0**Y * (1-p_0)**(1-Y) + (1-pi_0) * q_0**Y * (1-q_0)**(1-Y))
    pi=(1/n)*sum(mu)
    p=sum(Y*mu)/sum(mu)
    q=sum((1-mu)*Y)/sum(1-mu)
    if count%100==0:
        print(f"第{count}次迭代，估算参数分别为：pi={pi},p={p},q={q}")
    pi_0 = pi
    p_0 = p
    q_0 = q
    count+=1

#用拿到的估计参数重新模拟下
s=time.time()
print(f'二次模拟检验，模拟参数为pi={pi},p={p},q={q}……')
Y,pi,p,q=coin_experiment(pi, p, q, n,seed=21)
Y= np.array(Y)
print(f'模拟结束，共模拟{n}次，耗时{time.time()-s}……')
print(f"模拟结果，pi={pi},p={p},q={q}")



