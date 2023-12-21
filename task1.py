import numpy as np
import math
import time


def solve_lu(A, b):
    my_A = np.copy(A)
    n = my_A.shape[0]
    for k in range(n):
        for i in range(k+1,n):
            my_A[i][k] /= my_A[k][k]
            for j in range(k+1,n):
                my_A[i][j] -= my_A[i][k] * my_A[k][j]

    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.sum([my_A[i][j]*y[j] for j in range(i)])
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - np.sum([x[j]*my_A[i][j] for j in range(n-1,i-1,-1)]))/my_A[i][i]

    return x

def givenRotation(a, b):
    r = np.hypot(a, b)
    c = a / r
    s = b / r
    return c, s


def gmres(A, b, x0, eps):
    over = False
    n = len(b)
    p = n - 1
    c = np.zeros(n + 1)
    s = np.zeros(n + 1)
    h = np.zeros((n + 1, n))
    v = np.zeros((n + 1, n))

    r0 = b - np.dot(A, x0)
    beta = np.linalg.norm(r0)
    v[0] = r0 / beta

    e1 = np.zeros(n + 1)
    e1[0] = 1

    g = beta * e1

    for j in range(n):
        w = np.dot(A, v[j])

        for i in range(j + 1):
            h[i, j] = np.dot(w, v[i])
            w = w - h[i, j] * v[i]

        h[j + 1, j] = np.linalg.norm(w)

        if h[j + 1, j] == 0:
            p = j
            break

        v[j + 1] = w / h[j + 1, j]

        #  вращаем старыми коэффициентами
        for i in range(j):
            temp1 = c[i] * h[i, j] + s[i] * h[i + 1, j]
            temp2 = -s[i] * h[i, j] + c[i] * h[i + 1, j]
            h[i, j] = temp1
            h[i + 1, j] = temp2

        #  находим новые коэффициенты, зануляем последний элемент, предпоследний просто temp-ируем
        c[j], s[j] = givenRotation(h[j, j], h[j + 1, j])
        temp1 = c[j] * h[j, j] + s[j] * h[j + 1, j]
        temp2 = -s[j] * h[j, j] + c[j] * h[j + 1, j]
        h[j, j] = temp1
        h[j + 1, j] = temp2

        #  вращаем последними найденными коэффициентами вектор g
        temp1 = c[j] * g[j] + s[j] * g[j + 1]
        temp2 = -s[j] * g[j] + c[j] * g[j + 1]
        g[j] = temp1
        g[j + 1] = temp2

        if abs(g[j + 1]) < eps:
            over = True
            p = j
            break

    y = np.linalg.solve(h[:p + 1, :p + 1], g[:p + 1])
    x = x0 + np.dot(y, v[:p + 1])
    return x, over


def test():
    # testA = np.array([[1, 30], [2, 6]])
    # testB = np.array([3, -4])
    testA = np.zeros((50, 50))
    testB = np.random.randint(1, 51, size=50)
    for i in range(50):
        testA[i] = np.random.randint(1, 51, size=50)

    # Точное решение
    x_exact = np.linalg.solve(testA, testB)
    print("Точное решение:", x_exact)

    # Начальное приближение
    x_0 = np.zeros_like(testB)
    ans, metodOver = gmres(testA, testB, x_0, 10 ** (-11))
    max_iter = 50000
    iter = 1
    while not metodOver and iter < max_iter:
        ans, metodOver = gmres(testA, testB, ans, 10 ** (-4))
        iter += 1
    print(f'Final answer got after {iter:d} iterations')
    print(ans)


def main():
    eps = 10 ** (-11)
    for i in range(10, 50, 5):
        hh = math.pi / (i * 4)
        n = i
        print(f'Решаем систему с {n:d} кол-вом отрезков разбиений, с шагом {hh:f}')
        xs = np.linspace(0, math.pi / 4, n + 1)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        for j in range(2, n + 1):
            A[j - 1, j - 2] = -1 / hh ** 2
            A[j - 1, j] = -1 / hh ** 2
            A[j - 1, j - 1] = 2 / hh ** 2
        A[0, 0] = 1
        A[n, n] = 1
        for j in range(n + 1):
            b[j] = math.sin(xs[j])
        b[0] = 2
        b[n] = ((2 ** (1 / 2)) + 4) / 2

        x_exact = np.linalg.solve(A, b)
        # print("Точное решение:", x_exact)

        start_time = time.time()
        _ = solve_lu(A, b)
        end_time = time.time()
        exec_time = end_time - start_time
        print(f'Решение методом LU-разложений заняло {exec_time:f} секунд')

        # решаем систему методом GMRES
        start_time = time.time()
        x_0 = np.zeros_like(b)
        ans, metodOver = gmres(A, b, x_0, eps)
        max_iter = 25
        iter = 1
        while not metodOver and iter < max_iter:
            ans, metodOver = gmres(A, b, ans, eps)
            iter += 1
        end_time = time.time()
        exec_time =end_time - start_time
        print(f'Ответ с точностью {eps:f} получен после {iter:d} итераций, за {exec_time:f} секунд')


        # print('Вычисленное решение:', ans)
        gg = []
        for ii in range(len(A)):
            gg.append(abs(ans[ii] - x_exact[ii]))
        # print('Погрешность между точным и вычисленным решениями:', gg)
        print()


main()
