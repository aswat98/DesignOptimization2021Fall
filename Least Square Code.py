#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import torch as t
from torch.autograd import Variable

x1_info = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
p_info = [28.1,34.4,36.7,36.9,36.8,36.7,36.5,35.4,32.9,27.7,17.5]
p_wsat = 10.0**(8.07131-1730.63/(20.0+233.426))
p_dsat = 10.0**(7.43155-1554.679/(20.0+240.337))

def loss(a):
    total_loss = 0.0
    for i in range(11):
        x1 = x1_info[i]
        p = p_info[i]
        p_norm = x1*p_wsat*t.exp(a[0]*(a[1]*(1-x1)/(a[0]*x1+a[1]*(1-x1)))**2) + (1-x1)*p_dsat*t.exp(a[1]*(a[0]*x1/(a[0]*x1+a[1]*(1-x1)))**2)
        total_loss = total_loss + (p_norm-p)**2
    return total_loss

error = 1
A = Variable(t.tensor([1.0,1.0]), requires_grad = True)
while error>= 0.05:
    loss(A).backward()
    error = t.linalg.norm(A.grad)
    s = .2
    while loss(A-s*A.grad) > loss(A):
        s = .5*s
    with t.no_grad():
        A -= s*A.grad
        A.grad.zero_()
print(A)
print(loss(A))


# In[18]:


from math import exp
a = [1.9582,1.6893]
for i in range(11):
    x1 = x1_info[i]
    p_norm = x1*p_wsat*exp(a[0]*(a[1]*(1-x1)/(a[0]*x1+a[1]*(1-x1)))**2) + (1-x1)*p_dsat*exp(a[1]*(a[0]*x1/(a[0]*x1+a[1]*(1-x1)))**2)
    print((p_norm,3), p_info[i],((p_norm-p_info[i]/p_info[i],4)))


# In[ ]:





# In[ ]:




