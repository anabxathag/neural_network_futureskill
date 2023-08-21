x1 = [1.0,2.0,3.0,4.0,5.0]
y = [2.0,4.0,7.0,6.0,9.0]
w1 = 1.0
alpha = 0.05

def gradient(w1,x1,y):
    n = len(x1)
    del_w1 = 0.0
    for i in range(n):
        del_w1 += (w1*x1[i]-y[i])*x1[i]
    del_w1 = (2/n) * del_w1
    return del_w1

del_w1 = gradient(w1, x1, y)

while abs(del_w1) > 0.0001:
    w1_new = w1 - alpha * del_w1
    print("w1old = ", format(w1, '.6f'), ", del_w1 = ", format(del_w1, '.6f'),
          ", w1new = ", format(w1_new, '.6f'))
    w1 = w1_new
    del_w1 = gradient(w1, x1, y)
