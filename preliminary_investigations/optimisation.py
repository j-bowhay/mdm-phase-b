from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import dccp

n = 20
r = [1 for i in range(n)]

c = Variable(shape=(n,2))
constr = []
for i in range(n-1):
    for j in range(i+1,n):
        constr.append(norm(c[i,:]-c[j,:])>=r[i]+r[j])
prob = Problem(Minimize(max(max(abs(c),axis=1)+r)), constr)
prob.solve(method = 'dccp', ccp_times = 1)

l = max(max(abs(c),axis=1)+r).value*2
pi = np.pi
ratio = pi*sum(square(r)).value/square(l).value
print("ratio =", ratio)
print(prob.status)

# plot
fig, ax = plt.subplots()
circ = np.linspace(0,2*pi)
for i in range(n):
    ax.plot(c[i,0].value+r[i]*np.cos(circ),c[i,1].value+r[i]*np.sin(circ),'b')
ax.set_aspect('equal')
ax.set_xlim([-l/2,l/2])
ax.set_ylim([-l/2,l/2])
plt.show()