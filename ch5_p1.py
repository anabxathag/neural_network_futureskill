import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,1], [2,3], [4,2]], dtype=np.float64)
y = np.array([2,4,6], dtype=np.float64)
w = np.array([0,0], dtype=np.float64)
b = 0.0
alpha = 0.05

def gradient(w, b, x, y):
    n = len(x)

    del_w1 = (2/n) * np.matmul(np.matmul(x,w) + b - y, x.T[0])
    del_w2 = (2/n) * np.matmul(np.matmul(x,w) + b - y, x.T[1])
    del_b = (2/n) * np.matmul(np.matmul(x,w) + b - y, np.ones([n]))

    return del_w1, del_w2, del_b

del_w1, del_w2, del_b = gradient(w, b, x, y)

while (abs(del_w1) > 0.00001) or (abs(del_w2) > 0.00001) or (abs(del_b) > 0.00001) :
    w1_new = w[0] - alpha * del_w1
    w2_new = w[1] - alpha * del_w2
    b_new = b - alpha * del_b
    print("w1_new = ", format(w1_new, '.6f'),
          ", w2_new = ", format(w2_new, '.6f'),
          ", b_new = ", format(b_new, '.6f'))
    w[0] = w1_new
    w[1] = w2_new
    b = b_new
    del_w1, del_w2, del_b = gradient(w, b, x, y)
