import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы")
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


def functions(a_1, a_2):
    def find_extremum(a, b):
        if a == 0:
            return None
        return -b/(2*a)

    a11, a12, a13 = list(map(int, a_1.split())) 
    a21, a22, a23 = list(map(int, a_2.split()))

    extrema1 = find_extremum(a11, a12)
    extrema2 = find_extremum(a21, a22)

    a = a11 - a21
    b = a12 - a22
    c = a13 - a23
    if a == 0 and b == 0 and c==0:
        return None
    if a == 0 and b == 0:
       return []
    if a == 0:
       x = -c/b
       return [(x, a11 * x**2 + a12 * x + a13)]

    d = b*b - 4*a*c
    if d > 0:
        x1 = (-b + np.sqrt(d)) / (2*a)
        x2 = (-b - np.sqrt(d)) / (2*a)
        return [(x1, a11 * x1**2 + a12 * x1 + a13), (x2, a11 * x2**2 + a12 * x2 + a13)]
    if d == 0:
        x = -b/(2*a)
        return [(x, a11 * x**2 + a12 * x + a13)]
    return []


def skew(x):
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=0)
    m3 = np.mean((x - mean_x)**3)
    skewness = m3 / std_x**3
    return round(skewness, 2)

def kurtosis(x):
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=0)
    m4 = np.mean((x - mean_x)**4)
    kurt = m4 / std_x**4 - 3
    return round(kurt, 2)