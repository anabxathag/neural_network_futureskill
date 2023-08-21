import numpy as np

x = np.array([[1,1,1], [1,2,3], [1,4,2]], dtype=np.float64)
y = np.array([2,4,6], dtype=np.float64)
w = np.array([0,0,0], dtype=np.float64)
del_w = np.ones([3]) #initial del_w before enter the loop
alpha = 0.05
n = len(x)

while (max(abs(del_w)) > 0.00001):
    del_w = (2/n) * np.matmul(np.matmul(x,w) - y, x) #gradient learning
    w = w - (alpha * del_w)
    print("w = ", w)
