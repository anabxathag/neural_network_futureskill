import numpy as np

x = np.array([[1,1,1,-1,1,1,-1], [1,-1,1,1,-1,-1,1],
            [1,-1,-1,-1,1,1,-1], [1,1,-1,1,-1,-1,1],
            [1,1,1,-1,1,-1,1], [1,-1,-1,1,-1,1,-1],
            [1,-1,1,-1,1,-1,1], [1,1,-1,1,-1,-1,-1],
            [1,-1,1,-1,1,1,1], [1,1,-1,-1,1,1,-1]], dtype=np.float64)
y = np.array([-1,-1,1,-1,1,-1,1,-1,-1,1], dtype=np.float64)
w = np.zeros([7]) #initial weight 
del_w = np.ones([7]) #initial del_w
alpha = 0.05
n = len(x)

import math
def tanh(x): 
    return (math.e**x - math.e**(-x))/(math.e**x + math.e**(-x))

i = 0
while (max(abs(del_w)) > 0.00001):
    t_y_hat = tanh(np.matmul(x,w))
    del_w = (2/n) * np.matmul((t_y_hat - y)*(1-t_y_hat**2), x)
    w = w - (alpha * del_w)
    if (i%1000 == 0):
        print("w = ", w)
    i += 1

print("w = ", w)
