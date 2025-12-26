


from random import *
from math import gcd, log2
from time import *


def ro_pollard_log(p, a, b, r):

    def f(c, u, v, r):
        if c < p // 2:
            c = (a * c) % p
            u = (u + 1) % r
        else:
            c = (b * c) % p
            v = (v + 1) % r
        return c, u, v
    def f1(c, u, v, r):
    # Разбиение на 3 примерно равные части
        if c < p // 3:                          # Первая треть: умножаем на a (базу g)
            c = (a * c) % p
            u = (u + 1) % r                     # u — экспонента по a (g)
            # v не меняется
        elif c < 2 * (p // 3):                  # Вторая треть: возводим в квадрат
            c = (c * c) % p
            u = (u * 2) % r
            v = (v * 2) % r
        else:                                   # Последняя треть: умножаем на b (h = g^x)
            c = (b * c) % p
            v = (v + 1) % r
            # u не меняется
        return c, u, v
    
    # начальные значения
    uc = 2
    vc = 2
    ud = uc
    vd = vc

    c = ((a ** uc) * (b ** vc)) % p
    d = c

    # для отчета
    iterations = 0
    t1 = perf_counter()
    while (1):
        
        c, uc, vc = f1(c, uc, vc, r)

        d, ud, vd = f1(d, ud, vd, r)
        d, ud, vd = f1(d, ud, vd, r)

        # для отчета
        # с некоторой периодичностью будем выводить состояние. 
        # Если в течении пары часов не закончится - знаем на чем встали
        if iterations % 10 ** 8 < 5:
            if iterations % 10 ** 8 == 0:
                if iterations == 0:
                    print("Start situation")
                else:
                    print(f"\nCurrent situation. Working time: {round((perf_counter() - t1) / 60, 1)} мин.")
            print(f"i = {iterations // 10 ** 8} * 10^8 + {iterations % 10}: c = {c}, log_a(c) = {uc} + {vc}x. d = {d}, log_a(d) = {ud} + {vd}x")

        iterations += 1

        if (c == d):
            # решаем уравнение вида a + bx = c + dx (mod p)
            u = (uc - ud) % r
            v = (vd - vc) % r
            x = 0 

            # поиск x через НОД(v, r)
            g = gcd(v, r)
            if u % g != 0:
                return None

            v_ = v // g
            u_ = u // g
            r_ = r // g

            # "Переносим" множитель икса вправо
            x = (u_ * pow(v_, -1, r_)) % r_
            print(f"x = {x}, uc = {uc}, vc = {vc}, ud = {ud}, vd = {vd}, g = {g}")
            return x



def Gelfond(p, a, b, r):
    # отчет
    t1 = perf_counter()

    # 1
    m = int(r ** 0.5) + 1 # s в алгоритме

    # 2. L2[i] = b*a^(-i)mod p
    # находим a^{-1} mod p
    L1 = []
    ''' МЕДЛЕННЫЙ СПОСОБ
    for i in range(m):
        L1.append(b * pow(a, p - i - 1, p) % p)
    '''
    a_inv = pow(a, p - 2, p)   # a^{-1}
    cur = b
    for i in range(m):
        L1.append(cur)
        cur = (cur * a_inv) % p

    # для отчета
    print(f"time to build L1: {round((perf_counter() - t1) / 60, 1)} мин.")
    print(f"Первые элементы базы L1: {L1[0:5]}")
    print(f"Последние элементы базы L1: {L1[-5:]}")

    # 3
    L2 = []
    for j in range(m):
        L2.append(pow(a, m * j, p))
        #if (j % (10**5) == 0): print(f"Process: {round(j / m, 2)}")

        if L2[-1] in L1:
            ind = L1.index(L2[-1])
            x = (m * j + ind) % p
            print(f"Время работы: {round((perf_counter() - t1) / 60, 1)} мин.")

            return x, L1, L2, ind, j



def run_pollard(p, a, b, r):
    print(f"Расчетное время: {( (p ** 0.5) // 10**6)} сек.")
    print(f"x = {ro_pollard_log(p, a, b, r)}")

def run_gelfond(p, a, b, r):
    print(f"Расчетное время: {( (2 * ((r) ** 0.5) + log2((r))) // 10**6)} сек.")
    x, L1, L2, ind, j = Gelfond(p, a, b, p-1)

    print("Найденное значение x:", x)
    print(f"Значения: i (k) = {ind}, j (t) = {j}")
    print(f"Первые элементы базы L1: {L1[0:5]}")
    print(f"Последние элементы базы L1: {L1[-5:]}")
    print(f"Первые элементы базы L2: {L2[0:5]}")
    print(f"Последние элементы базы L2: {L2[-5:]}")


p = 304785986349532418519
a = 3
b = 19
r = 152392993174766209259
p = 1000000000039   # простое
a = 2
x = 987654321
b = pow(a, x, p)
print("1. Метод ро-Полларда")
print("2. Метод Гельфонда")
c = int(input())
if c == 1:
    run_pollard(p, a, b, r)
elif c == 2:
    run_gelfond(p, a, b, r)





'''

p = 1000000000039   # простое
a = 2
x = 987654321
b = pow(a, x, p)
'''