#from math import *
from time import *
from random import *
from math import log
import math
import time
from itertools import combinations

def gcd(x, y):
    return math.gcd(x, y)

def f(x):
    const = 1
    func = (x * x + const)
    return func

def time_ro_pollard(time, n, iterations):
    result_time = (n ** 0.25) * (time / iterations) - 0.1 # 0.1 - затраты на подсчет времени
    print(f"Ожидаемое время выполнения: {round(result_time, 2)} сек")

def ro_pollard(n, c = 1):
    # 0 для подсчета времени:
    iterations_to_measure_time = 1000
    t1 = perf_counter()
    t2 = 0
    iterations = 0
    # 1
    a = [c]
    b = [c]
    d = []

    while (1):

        #0.5
        iterations += 1
        if (iterations_to_measure_time) > 0: 
            t2 = perf_counter()
            iterations_to_measure_time -= 1
            if (iterations_to_measure_time == 0): time_ro_pollard(t2 - t1, n, 1000)

        # 2
        a.append(f(a[-1]) % n)
        b.append(f(f(b[-1]) % n) % n)
        #b.append(f(b[-1]) % n)

        # 3
        d.append(gcd(a[-1] - b[-1], n))

        # 4
        if (1 < d[-1] < n):
            return a, b, d
        elif (d[-1] == n): return 0 # делитель не найден
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

def choose_B_size(n):
    if n < 10 ** 20: 
        return int(n ** 0.2) # для второго числа. Эмпирическим путем лучшее значение около n ** 0.2
    if n < 10 ** 40: 
        return int(n ** 0.15) # для второго числа. Эмпирическим путем лучшее значение около n ** 0.15
    if n < 10 ** 60: 
        return 3000000
    if n < 10 ** 90: 
        return 20000000
    else:
        print("Число слишком большое")
        exit()

def expected_time(n, P):
    print(f"Ожидаемое время работы: {round((P * log(P) * (log(n) ** 2)) / (10 ** 10 * 2), 2)} секунд")

def ro_m1_pollard(n):
    # 1
    B_size = choose_B_size(n)
    B = get_primes(B_size) 
    a = []

    # для отчета
    a_src = []
    iterations = 0
    expected_time(n, B[-1])
    # 2
    while (1):
        
        a.append(randint(2, n - 2))
        a_src.append(a[-1]) # для отчета
        d = gcd(a[-1], n)
        if d >= 2: return d

        # 3
        l = []
        log_n = math.log(n)
        for pi in B:
            l.append(int(log_n / math.log(pi)))
            a.append(pow(a[-1], pow(pi, l[-1]), n))

            # 3.5 добавим условие остановки
            d = gcd(a[-1] - 1, n)
            if (d != 1 and d != n): break

        iterations += 1
        # 4
        d = gcd(a[-1] - 1, n)
        if (d == 1 or d == n): pass#return 0
        else: return d, a_src, l, B, iterations


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
    else: e[0] = 0
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

def get_Pi(n, count, B):

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
    e = []

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
    
    # генерируем следующие a_k и P_k
    for i in range(1, count):
        # шаг разложения sqrt(n)
        m = d * a - m
        d = (n - m*m) // d
        a = (a0 + m) // d

        # рекурсия для P_k
        Pk = a * P0 + P_minus1

        Pis.append(Pk) # если числитель гладкий - добавляем
        # проверяем на гладкость сразу, чтоб не добавлять лишнего
        #e_i = smooth_factor(Pk, n, B) # e_i
        #if (e_i): 
        #    Pis.append(Pk) # если числитель гладкий - добавляем
        #    e.append(e_i)
             

        # сдвиг
        P_minus1, P_minus2 = P0, P_minus1
        P0 = Pk
        if (i == count // 10): print("good")

    return Pis

import math

def get_Pi_collect(n, count, B, target_relations=None):
    """
    Генерирует числители Pk для sqrt(n), но сохраняет только B-гладкие.
    Возвращает (Pis, e), где Pis[i] соответствует e[i].
    count - максимальное количество шагов цепной дроби (безопасный лимит).
    target_relations - если задано, остановится, когда найдено target_relations реляций.
    """
    a0 = int(math.isqrt(n))
    m = 0
    d = 1
    a = a0

    P_minus2 = 0
    P_minus1 = 1
    P0 = a0  # первый числитель

    Pis = []
    e = []

    # Проверим P0 на гладкость сразу (если хочется включить a0)
    e0 = smooth_factor(P0, n, B)
    if e0:
        Pis.append(P0)
        e.append(e0)
        if target_relations and len(Pis) >= target_relations:
            return Pis, e

    for i in range(1, count):
        # шаг непрерывной дроби
        m = d * a - m
        d = (n - m*m) // d
        a = (a0 + m) // d

        Pk = a * P0 + P_minus1

        # проверяем на гладкость сразу
        e_i = smooth_factor(Pk, n, B)
        if e_i:
            Pis.append(Pk)
            e.append(e_i)
            if target_relations and len(Pis) >= target_relations:
                return Pis, e

        # сдвиг
        P_minus1, P_minus2 = P0, P_minus1
        P0 = Pk
        if (i % 1000) == 0: 
            print((i / count))
    return Pis, e

#Находит все линейные зависимости между строками матрицы mod 2.
#Возвращает список векторов зависимостей.
import numpy as np

def gauss_mod2_numpy(matrix):
    m = np.array(matrix, dtype=np.uint8)
    rows, cols = m.shape
    trans = np.eye(rows, dtype=np.uint8)

    r = 0
    for c in range(cols):
        # pivot
        pivots = np.where(m[r:, c] == 1)[0]
        if len(pivots) == 0:
            continue
        pivot = pivots[0] + r

        # swap
        m[[r, pivot]] = m[[pivot, r]]
        trans[[r, pivot]] = trans[[pivot, r]]

        # eliminate
        other = np.where((m[:, c] == 1) & (np.arange(rows) != r))[0]
        for i in other:
            m[i] ^= m[r]
            trans[i] ^= trans[r]

        r += 1

    # rows of zeros -> dependencies rows
    deps = [trans[i].tolist() for i in range(rows) if not m[i].any()]
    return deps

    rows = [bitarray(row) for row in matrix]
    n = len(rows)
    trans = [bitarray('0')*n for _ in range(n)]
    for i in range(n):
        trans[i][i] = 1

    r = 0
    cols = len(rows[0])

    for c in range(cols):
        pivot = None
        for i in range(r, n):
            if rows[i][c]:
                pivot = i
                break
        if pivot is None:
            continue

        rows[r], rows[pivot] = rows[pivot], rows[r]
        trans[r], trans[pivot] = trans[pivot], trans[r]

        for i in range(n):
            if i != r and rows[i][c]:
                rows[i] ^= rows[r]
                trans[i] ^= trans[r]
        r += 1

    return [trans[i].tolist() for i in range(n) if not rows[i].any()]

def cont_fract(n):
    # func variables
    
    primes_for_B = int(2.72 ** ((2 / 3) * math.sqrt(log(n) * log(log(n)))))
    num_of_generated_numerators = primes_for_B + 1 # гарантированное значение, чтоб появилась линейная зависимость в e
    # 1
    B = [-1, 2]
    B_tmp = (get_primes(primes_for_B))[1:] # заготовка для B, со всеми простыми числами до 1009, кроме 2 (т.к. для 2 якоби нельзя)
    for b in B_tmp:
        if yakobi(n, b) == 1: B.append(b)
    
    h = len(B) - 1 # убираем -1 в начале

    # 2 - находим из них h+2 B-гладких числа
    P, e = get_Pi_collect(n, num_of_generated_numerators, B) # генерируем числители подходящих дробей

    # проверка на ei ⊕ ej ⊕ ... == 0
    e_mod2 = [[v % 2 for v in row] for row in e]

    print("start gauss")
    dependencies = gauss_mod2_numpy(e_mod2)
    print("end gauss")

    s = 0
    t = 0
    for d in dependencies:
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

            t *= (B[i] ** degree)

        if((s % n) == (t % n)) or ((s % n) == ((-t) % n)):
            continue  # плохой случай, вернуться искать дальше
        break
    
    q = gcd(s - t, n)

    return B, P, e, s % n, t % n

    #print(e)

def ro_pollard_show(n):
    t1 = perf_counter()
    a, b, d = ro_pollard(n)
    if (len(a) < 1000): print("Ожидаемое время работы: ваще быстра")
    t2 = perf_counter()
    print("АЛГОРИТ ρ-ПОЛЛАРДА:")
    print("Первые и последние 5 значений:")
    print("a:", a[0:5], a[-5:])
    print("b:", b[0:5], b[-5:])
    print("НОД(a - b, n):", d[0:5], d[-5:])

    print("Время работы:", round(t2 - t1, 2), "сек")
    print("Разложение числа:", n, "=", d[-1], "*", n // d[-1])
    print()

def ro_m1_pollard_show(n):
    t1 = perf_counter()
    d, a, l, B, iterations = ro_m1_pollard(n)
    t2 = perf_counter()
    print("АЛГОРИТ ρ-1 ПОЛЛАРДА:")
    print(f"Значения показателей l_i {l[0:10]}, ..., {l[-11:]}")
    print(f"Использованные основания а: {a}")
    print(f"Основание а, при котором выполнено разложение: {a[-1]}")
    print(f"База разложения B: простые числа от {B[0]} до {B[-1]}")
    print("Итераций:", iterations)
    print("Время работы:", round(t2 - t1, 2), "сек")

    print(f"Разложение числа: {n} = {d} * {n // d}")
    print()

def cont_fract_show(n):
    print("Ожидаемое время выполнения:", round((2.72 ** (2 *(log(n) * log((log(n)))) ** 0.5)) / 10 ** 12, 2), "сек")
    t1 = perf_counter()
    B, P, e, s, t = cont_fract(n)
    t2 = perf_counter()
    Pi2 = []
    for i in range(len(P)):
        Pi2.append((P[i] * P[i]) % n)
    answer = gcd(s - t, n)

    print("МЕТОД НЕПРЕРЫВНЫХ ДРОБЕЙ:")
    if (len(B) > 10):
        print(f"Кол-во элементов базы разложения: {len(B)}")
        print("Последний элемент базы разложения:", B[-1])
    else: print(f"База разложения: {B}")

    print("Значения числителей подходящих дробей (5 первых):", P[0:5])
    print("Соответсвующие B-гладкие значения (P_i)^2 (mod n) (5 первых):", Pi2[0:5])
    #print("Векторы показателей для B-гладких значений (5 первых):", *e[0:5], sep='\n')
    print("Метод, использованный для поиска линейно-зависимых векторов: Гауссова-исключения")
    print(f"s = {s}, t = {t}")
    print(f"Разложение числа: {n} = {answer} * {n // answer}")
    print("Время работы:", round(t2 - t1, 2), "сек")


print("Выберите число для разложения:")
print("1.", 46196147461358036447)
print("2.", 817163034319287615259456824822306714767)
print("3.", 10979553153460790109129491311200000000009346725710714598774710806234346092586611)
choose = int(input())
num = 0
if choose == 1:
    num = 46196147461358036447
elif choose == 2:
    num = 817163034319287615259456824822306714767
elif choose == 3:
    num = 10979553153460790109129491311200000000009346725710714598774710806234346092586611
else: exit()
#num = 21299881
#ro_pollard_show(num)
#ro_m1_pollard_show(num)
#a = pollard_p_minus_1(817163034319287615259456824822306714767)
cont_fract_show(num)

'''
21299881 = 5531 * 
46196147461358036447 = 7413206131 * 6231601637
817163034319287615259456824822306714767 = 33528055879943520859 * 24372514685771400413
'''
