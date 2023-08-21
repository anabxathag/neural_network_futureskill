import numpy as np

#Stock raw data
raw = [16,15.7,17,16.9,17.1,17,16.4,16.7,16.4,17,16.9,17,17,
       16.9,16.5,17.2,17.2,16.9,17,16.8,16.6,16.4,16.4,15.8,15.9,
       15.8,15.6,15.5,15.7,15.5,15.4,15.5,15.5,15.3,15.4,15.3,15.3,
       15.4,15.3,15.3,15.4,15.3,15.4,15.8,15.6,15.6,15.7,15.4,15.6,
       15.4,14.7,15.1,15.7,15.7,15.6,15.4,15.5,15.7,15.4,15.6,15.6,
       15.3,15.4,15,14.7,13,12.8,12.8,12.8,13,12.7,13,13.2,13.7,13.7,
       13.4,13.0,12.9,13,13,12.9,12.7,12.8,13,12.7,11.8,11.9,12.1,
       12.1,12,11.9,12.1,12,12.1,12.3,12.3,12.1,11.7,11.8,11.5,
       11.8,11.8,12,12.3,12.8,12.9,13.6,13.3,12.7,13,13,12.9,11.9,
       11.7,11.9,11.6,11,10.5,10.7,11,10.9,10.9,10.9,11,10.8]
num = len(raw)
print("data length = ", num)
print(raw)

raw2 = []
for i in range(num-1):
    if raw[i] > raw[i+1]:
        raw2.append(-1)
    elif raw[i] < raw[i+1]:
        raw2.append(1)
    else:
        raw2.append(0)
num = len(raw2)
print("data length = ", num)
print(raw2)
raw = raw2

# Split Data
x_raw = []
y_raw = []
for i in range(num-4):
    new_row_x = [1, raw[i], raw[i+1], raw[i+2], raw[i+3]]
    new_row_y = raw[i+4]
    x_raw.append(new_row_x)
    y_raw.append(new_row_y)
    print(new_row_x, " ", new_row_y)

x = np.array(x_raw, dtype=np.float64)
y = np.array(y_raw, dtype=np.float64)
w = np.zeros([5]) #initial weight 
del_w = np.ones([5]) #initial del_w
alpha = 0.05
n = len(x)

import math
def tanh(x): 
    return (math.e**x - math.e**(-x))/(math.e**x + math.e**(-x))
    
i = 0
while (max(abs(del_w)) > 0.00000001):
    t_y_hat = tanh(np.matmul(x,w))
    del_w = (2/n) * np.matmul((t_y_hat - y)*(1-t_y_hat**2), x)
    w = w - (alpha * del_w)
    if (i%1000 == 0):
        print("w = ", w)
    i += 1

print("w = ", w)

# predict
for i in range(n):
    print("x = ", x[i], " y = ", y[i], " y_hat = ", np.matmul(x[i],w))


