

from time import*

def evklid_adv(a, b):

    # проверяем входные данные
    if (a <= 0) or (b <= 0):
        print("numbers must be greater than zero")
    
    # устанавливаем начальные значения
    x = [1, 0]
    y = [0, 1]
    r = [-1 for i in range(1000)] 
    q = [0]
    i = 1

    if a >= b:
        r[0] = a
        r[1] = b
    else:
        r[0] = b
        r[1] = a
    
    while (1):
        r[i + 1] = r[i - 1] % r[i]
        if (r[i + 1] == 0): break

        # q[i]
        q.append(r[i - 1] // r[i])
        #x[i + 1]
        x.append(x[i - 1] - q[i] * x[i])
        #y[i + 1]
        y.append(y[i - 1] - q[i] * y[i])

        i += 1

    print("Итераций произведено: ", i)
    print("Расширенный алгоритм Евклида:")
    print("Результат: НОД = ", r[i])

    print("x[0 : 5]:", x[0:5])
    print("x[i-5:i]:", x[i - 4:i + 1])
    print("y[0 : 5]:", y[0:5])
    print("y[i-5:i]:", y[i - 4:i + 1])
    #print("q[0 : 5]:", q[0:5])
    #print("q[i-5:i]:", q[i - 5:i + 1])
    print("r[0 : 5]:", r[0:5])
    print("r[i-4:i+2]:", r[i - 4:i + 2])
    
    print()


def bin_evklid(a, b):
    # проверяем входные данные
    if (a <= 0) or (b <= 0):
        print("numbers must be greater than zero")

    g = 1

    while (a % 2 == 0 and b % 2 == 0):
        a //= 2
        b //= 2
        g *= 2

    u = [a]
    v = [b]
    A = [1]
    B = [0]
    C = [0]
    D = [1]
    x = []
    y = []

    while (u[-1] != 0):
        while (u[-1] % 2 == 0):
            u.append(u[-1] // 2)
            # при каждеом изменении u или v записываем соответствующие им А и В, C и D
            A.append(A[-1])
            B.append(B[-1])
            if (A[-1] % 2 == 0) and (B[-1] % 2 == 0):
                A[-1] = A[-1] // 2
                B[-1] = B[-1] // 2
            else:
                A[-1] = (A[-1] + b) // 2
                B[-1] = (B[-1] - a) // 2

        while (v[-1] % 2 == 0):
            v.append(v[-1] // 2)
            C.append(C[-1])
            D.append(D[-1])
            if (C[-1] % 2 == 0) and (D[-1] % 2 == 0):
                C[-1] = C[-1] // 2
                D[-1] = D[-1] // 2
            else:
                C[-1] = (C[-1] + b) // 2
                D[-1] = (D[-1] - a) // 2

        if (u[-1] >= v[-1]):
            u.append(u[-1] - v[-1])
            A.append(A[-1] - C[-1])
            B.append(B[-1] - D[-1])
        else: 
            v.append(v[-1] - u[-1])
            C.append(C[-1] - A[-1])
            D.append(D[-1] - B[-1])



    d = int(g * v[-1])


    print("Бинарный алгоритм Евклида:")
    print("Результат: НОД = ", d)
    print("len(u): ", len(u))
    print("u1: ", u[0:5])
    print("u2: ", u[-5:-1], u[-1])
    print("A1: ", A[0:5])
    print("A2: ", A[-5:-1], A[-1])
    print("B1: ", B[0:5])
    print("B2: ", B[-5:-1], B[-1])
    print()
    print("len(v): ", len(v))
    print("v1: ", v[0:5])
    print("v2: ", v[-5:-1], v[-1])
    print("C1: ", C[0:5])
    print("C2: ", C[-5:-1], C[-1])
    print("D1: ", D[0:5])
    print("D2: ", D[-5:-1], D[-1])

    
    

# с усеченными остатками
def evklid_adv_pro_max_smartfon_vivo(a, b):
    # проверяем входные данные
    if (a <= 0) or (b <= 0):
        print("numbers must be greater than zero")
    
    if (b > a): 
        a, b = b, a

    # устанавливаем начальные значения
    
    x = [0 for i in range(1000)]
    x[0] = 1
    x[1] = 0
    y = [0 for i in range(1000)]
    y[0] = 0
    y[1] = 1
    
    r = [] 
    q = []
    i = 0

    while (b != 0):
        # усеченное частное
        q.append((a + b // 2) // b)

        r.append(a - q[i] * b)
        a, b = b, r[i]

        x[i + 1], x[i + 2] = x[i + 1], x[i] - q[i] * x[i + 1]
        y[i + 1], y[i + 2] = y[i + 1], y[i] - q[i] * y[i + 1]
        
        i += 1
    a = abs(a)

    print("Расширенный алгоритм Евклида с усеченными остатками:")
    print("Результат: НОД = ", a)
    print("Итераций: ", i)
    print("x[0 : 5]:", x[0:5])
    print("x[i-5:i]:", x[i - 4:i + 1])
    print("y[0 : 5]:", y[0:5])
    print("y[i-5:i]:", y[i - 4:i + 1])
    #print("q[0 : 5]:", q[0:5])
    #print("q[i-5:i]:", q[i - 5:i + 1])
    print("r[0 : 5]:", r[0:5])
    print("r[i-4:i+2]:", r[i - 5:i + 2])


# main

a = 19387207028695801417
b = 19387226047544992493


t1 = time()
evklid_adv(a, b)
t2 = time()
print("Время работы (сек): ", t2 - t1)
print()

t1 = time()
bin_evklid(a, b)
t2 = time()
print("Время работы (сек): ", t2 - t1)
print()

t1 = time()
evklid_adv_pro_max_smartfon_vivo(a, b)
t2 = time()
print("Время работы (сек): ", t2 - t1)

