import numpy as np

x = np.array([[1,1,1,0,1,1,0], [1,0,1,1,0,0,1],
            [1,0,0,0,1,1,0], [1,1,0,1,0,0,1],
            [1,1,1,0,1,0,1], [1,0,0,1,0,1,0],
            [1,0,1,0,1,0,1], [1,1,0,1,0,0,0],
            [1,0,1,0,1,1,1], [1,1,0,0,1,1,0]], dtype=np.float64)
y = np.array([0,0,1,0,1,0,1,0,0,1], dtype=np.float64)
w = np.zeros([7]) #initial weight 
del_w = np.ones([7]) #initial del_w
alpha = 0.05
n = len(x)

import math
def sigmoid(x): 
    return (math.e**x)/(1+math.e**x)
    
i = 0
while (max(abs(del_w)) > 0.0001):
    s_y_hat = sigmoid(np.matmul(x,w))
    del_w = (2/n) * np.matmul((s_y_hat - y)*s_y_hat*(1-s_y_hat), x)
    w = w - (alpha * del_w)
    if (i%1000 == 0):
        print("w = ", w)
    i += 1

print("w = ", w)
