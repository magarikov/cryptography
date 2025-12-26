
from time import *


a = 1282535
b = 412
n = 9998887776
t1 = time()
for i in range(n):
    i += 1
    v = pow(a, b) * i
    t2 = time()
    if (i % 10000 == 0):
        print(v % 100)
        print("                      ", i, t2 - t1)