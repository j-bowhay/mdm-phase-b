from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp

n = 2

d = Variable()
c = Variable(shape=(n,2))
constr = []
for i in range(n-1):
    for j in range(i+1,n):
        constr.append(norm(c[i,:]-c[j,:])>=d)

prob = Problem(Maximize(d), constr + [0 <= c, c<=1 ])

print(dccp.is_dccp(prob))

prob.solve(method='dccp',ccp_times=10,max_iter=1000)