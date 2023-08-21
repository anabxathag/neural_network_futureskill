import numpy as np

x = np.array([[1,1,1,1,1,1], [1,2,2,2,2,2],
              [1,3,3,3,3,3], [1,4,4,4,4,4], [1,5,5,5,5,5]], dtype=np.float64)
y = np.array([2,4,7,6,9], dtype=np.float64)
w = np.zeros([6]) #initial weight 
del_w = np.ones([6]) #initial del_w
alpha = 0.005
n = len(x)

while (max(abs(del_w)) > 0.00001):
    del_w = (2/n) * np.matmul(np.matmul(x,w) - y, x) #gradient learning
    w = w - (alpha * del_w)
    print("w = ", w)
