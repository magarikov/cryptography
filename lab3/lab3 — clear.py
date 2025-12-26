#from math import *
from time import *
from random import *
import math
import time
from itertools import combinations

import random

def pollard_rho(n, c=1, par=5):
    def f(x):
        return (x*x + par) % n

    a, b, d, i = c, c, 1, 0
    output = []

    t0 = time.perf_counter()
    while True:
        i += 1
        a = f(a)
        b = f(f(b))
        d = math.gcd(abs(a - b), n)
        output.append(f"{i:>3} |  {a:>80}   {b:>80}    {d:>10}")
        if i == 10000:
            t1 = time.perf_counter()
            calc_time = (n**0.25) / 10000 * (t1 - t0)
            print(f"Вычисленное примерное время: {calc_time:.5f} секунд")
            if calc_time > 3600:
                print(f"Расчётное время больше часа")
                return output

        if 1 < d < n:
            return output
        if d == n:
            return None

def gcd(x, y):
    return math.gcd(x, y)

def f(x):
    const = 1
    func = (x * x + const)
    return func

def ro_pollard(n, c = 1):
    t1 = perf_counter()
    # 1
    a = c
    b = c

    while (1):
        # 2
        a = f(a) % n
        b = f(b) % n
        b = f(b) % n

        # 3
        d = gcd(a - b, n)

        # 4
        if (1 < d < n):
            t2 = perf_counter()
            print(t2 - t1)
            return d
        elif (d == n): return 0 # делитель не найден
        # если d == 1, продолжаем цикл


def get_primes(limit):  # сложность O(nlog(log(n)))
    primes = []
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for p in range(2, limit + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    return primes


def ro_m1_pollard(n):
    # 1
    B = get_primes(7019801) # для второго числа

    # 2
    for j in range(1000):
        a = randint(2, n - 2)
        d = gcd(a, n)
        if d >= 2: return d

        # 3
        log_n = math.log(n)
        for pi in B:
            l = int(log_n / math.log(pi))
            a = pow(a, pow(pi, l), n)

            # 3.5 добавим условие остановки
            d = gcd(a - 1, n)
            if (d != 1 and d != n): break
        
        # 4
        d = gcd(a - 1, n)
        if (d == 1 or d == n): pass#return 0
        else: return d



# Алгоритм итеративно:
#Сводит a к a mod n.
#Убирает все степени 2 из a (пишем a = 2^k * a1).
#Вносит множитель от 2^k в знак s (формула через n % 8).
#Если a1 == 1 — возвращает накопленный знак.
#Применяет закон квадратичной реципроцности: меняем (a1/n) на (n mod a1)/a1, возможно меняя знак, 
# если оба ≡ 3 (mod 4).
#Повторяем.
#Если в какой-то момент a == 0 — возвращаем 0 (если n>1)

# берем для нахождения символа Лежандра для построения B в cont_fract
def yakobi(a, n):

    if (n < 3):
        print("n must be > 3")
    #if (a < 0) or (a > n):
    #    print("a must be: 0 <= a <= n")

    # учитывает знак от (2/n):
    # = 1, n ≡ 1,7 (mod 8)
    # = -1, n ≡ 3,5 (mod 8)
    s = 0
    
    # 1
    # накопитель общего знака
    g = 1

    while (1):
        
        # 2-3
        if a == 0:
            return 0
        if a == 1:
            return g
        
        # 4
        k = 0
        while (a % 2 == 0):
            k += 1
            a = a // 2
        a1 = a

        # 5
        if k % 2 == 0:
            s = 1
        else:
            if (n % 8 == 1) or (n % 8 == 7):
                s = 1
            if (n % 8 == 3) or (n % 8 == 5):
                s = -1

        # 6
        if a1 == 1:
            return g * s
        
        # 7 
        if (n % 4 == 3) and (a1 % 4 == 3):
            s = -s
        
        # 8
        a = n % a1
        n = a1
        g = g * s

# Проверка B-гладкости и построение вектора показателей
#def smooth_factor(n, base, number_of_digits):
    exponents = [0] * number_of_digits
    start = 0
    if base[0] == -1:
        start = 1
        if n < 0:
            exponents[0] = 1
            n *= -1

    for i in range(start, number_of_digits):
        p = base[i]
        if n % p == 0:
            e = 1
            n = n // p
            while n % p == 0:
                n = n // p
                e += 1
            exponents[i] = e
        if n == 1:
            return exponents
    return False

# 221 стр в книжке. Число B-гладкое, если в B содержатся простые числа, перемножением которых можно получить это число
def smooth_factor(Pi, n, B):
    Pi2 = (Pi * Pi) % n  # Pi^2 (mod n)
    if (abs(Pi2) > abs(Pi2 - n)): Pi2 = Pi2 - n # поправка, т.к. Python при "%" не даёт отрицательных чисел, а нам надо
    e = [] # вектор используемых чисел. Если на i-ом индексе 1, значит i-ое число из B используется в разложении
    for _ in range(len(B)): e.append(0) # заполяем нулями 

    # заранее обрабатываем отрицательные числа
    if Pi2 < 0:
        Pi2 = -Pi2
        e[0] = 1
    else: e.append(0)
    # после этого убираем "-1" из B
    B = B[1:]
    
    # пытаемся разложить число, используя только множители из B (критерий гладкости)
    for i in range(len(B)):
        while (Pi2 % B[i] == 0):
            Pi2 = Pi2 // B[i]
            e[i + 1] += 1 # добавляем степень (i+1, т.к. 1ый эл-т в B убиран)

    # если получилось разложить - возвращаем вектор разложения
    if Pi2 == 1: return e
    else: return False


def get_Pi(n, count):
    # разложение sqrt(n) в непрерывную дробь
    a0 = int(math.isqrt(n))
    m = 0
    d = 1
    a = a0

    # начальные P
    P_minus2 = 0
    P_minus1 = 1
    P0 = a0  # первый числитель

    Pis = [P0]

    # генерируем следующие a_k и P_k
    for _ in range(1, count):
        # шаг разложения sqrt(n)
        m = d * a - m
        d = (n - m*m) // d
        a = (a0 + m) // d

        # рекурсия для P_k
        Pk = a * P0 + P_minus1

        Pis.append(Pk)

        # сдвиг
        P_minus1, P_minus2 = P0, P_minus1
        P0 = Pk

    return Pis

def gaussian_elimination_mod2(matrix):
    """
    Находит все линейные зависимости между строками матрицы mod 2.
    Возвращает список векторов зависимостей.
    """

    m = [row[:] for row in matrix]  # копия
    rows = len(m)
    cols = len(m[0])
    
    # Матрица преобразований, начинаем как единичную
    trans = [[1 if i == j else 0 for j in range(rows)] for i in range(rows)]

    r = 0  # текущая строка

    for c in range(cols):
        # находим строку с 1 в позиции (>= r)
        pivot = None
        for i in range(r, rows):
            if m[i][c] == 1:
                pivot = i
                break

        if pivot is None:
            continue

        # меняем местами строки
        m[r], m[pivot] = m[pivot], m[r]
        trans[r], trans[pivot] = trans[pivot], trans[r]

        # обнуляем остальные строки
        for i in range(rows):
            if i != r and m[i][c] == 1:
                # XOR строки
                for j in range(cols):
                    m[i][j] ^= m[r][j]
                for j in range(rows):
                    trans[i][j] ^= trans[r][j]

        r += 1

    # теперь строки, у которых всё нули — это свободные строки
    dependencies = []
    for i in range(rows):
        if all(v == 0 for v in m[i]):
            dependencies.append(trans[i])

    return dependencies

def gauss_elimination(exponents):
    num_rows = len(exponents)
    if num_rows == 0:
        return []
    num_cols = len(exponents[0])
    reduced = []
    for vector in exponents:
        value = 0
        for i in range(num_cols):
            if vector[i] % 2 != 0:
                value += 1 << (num_cols - i - 1)
        reduced.append(value)

    history = []
    value = 1 << (num_rows - 1)
    for _ in range(num_rows):
        history.append(value)
        value >>= 1

    bit = 1
    max_size = 1 << (num_cols - 1)
    while bit <= max_size:
        pivot_found = False
        for i in range(num_rows):
            entry = reduced[i]
            if entry & bit and entry % bit == 0:
                pivot_found = True
                his = history[i]
                break

        if pivot_found:
            for m in range(i + 1, num_rows):
                if reduced[m] & bit:
                    reduced[m] ^= entry
                    history[m] ^= his
        bit <<= 1

    vectors = []
    for i in range(num_rows):
        value = history[i]
        if reduced[i] != 0 or value == 0:
            continue
        vector = [0] * num_rows
        j = num_rows - 1
        while j >= 0:
            if value % 2:
                vector[j] = 1
            value >>= 1
            j -= 1
        vectors.append(vector)
    return vectors




def cont_fract(n):
    # func variables
    primes_for_B = 1000
    num_of_generated_numerators = 300000#int(2.72 ** ((2 / 3) * math.sqrt(math.log(n) * math.log(math.log(n)))))#5000

    # 1
    B = [-1, 2]
    B_tmp = (get_primes(primes_for_B))[1:] # заготовка для B, со всеми простыми числами до 1009, кроме 2 (т.к. для 2 якоби нельзя)
    for b in B_tmp:
        if yakobi(n, b) == 1: B.append(b)
    
    h = len(B) - 1 # убираем -1 в начале

    # 2 - находим из них h+2 B-гладких числа
    P_tmp = get_Pi(n, num_of_generated_numerators) # генерируем числители подходящих дробей
    P = [] # сюда запишем только гладкие числители
    e = [] # матрица векторов для гладких Pi. если a - степень использованного из B числа для Pi, то e_i = a % 2. РЕАЛЬНО ХРАНИМ a, А "%" СЧИТАЕМ В МОМЕНТЕ
    for i in P_tmp:
        e_i = smooth_factor(i, n, B) # e_i
        if (e_i): 
            P.append(i) # если числитель гладкий - добавляем
            e.append(e_i)
        
    index_history = []

    # проверка на ei ⊕ ej ⊕ ... == 0
    e_mod2 = [[v % 2 for v in row] for row in e]

    deps = gaussian_elimination_mod2(e_mod2)

    for d in deps:
        indexes = [i for i in range(len(d)) if d[i] == 1]

        s = 1
        for i in range(len(P)):
            if i in indexes: 
                s *= P[i]
                
        s = s % n
        t = 1
        for i in range(len(B)):
            degree = 0
            for j in range(len(e)):
                if j in indexes:
                    degree += e[j][i]
            degree = degree // 2

            t *= B[i] ** degree

        if((s % n) == (t % n)) or ((s % n) == ((-t) % n)):
            continue  # плохой случай, вернуться искать дальше
        break
    
    q = gcd(s - t, n)

    print(n / q)

    #print(e)




#num = 21299881
#num = 46196147461358036447
num = 817163034319287615259456824822306714767
#num = 10979553153460790109129491311200000000009346725710714598774710806234346092586611
t1 = perf_counter()
#a = ro_m1_pollard(817163034319287615259456824822306714767)
#a = pollard_p_minus_1(817163034319287615259456824822306714767)
a = cont_fract(num)
t2 = perf_counter()
print(a, t2 - t1)

'''
21299881 = 5531 * 
46196147461358036447 = 7413206131 * 6231601637
817163034319287615259456824822306714767 = 
'''