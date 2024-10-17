import numpy as np
import math


def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Количество столбцов первой матрицы не равно числу строк во второй матрицы.")
    
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]
    
    for i in range(len(matrix_a)): 
        for k in range(len(matrix_b[0])): 
            for j in range(len(matrix_b)):  
                result[i][k] += matrix_a[i][j] * matrix_b[j][k]
    
    return result


def find_extremum(a, b):
    if a == 0:
        return None
    return -b / (2 * a)

def functions(a_1, a_2):    
    a11, a12, a13 = map(int, a_1.split())
    a21, a22, a23 = map(int, a_2.split())

    extrema1 = find_extremum(a11, a12)
    extrema2 = find_extremum(a21, a22)

    a = a11 - a21
    b = a12 - a22
    c = a13 - a23

    if a == 0 and b == 0 and c == 0:
        return None  # Бесконечное количество решений

    if a == 0:
        if b == 0:
            return []  # Нет решений
        x = -c / b  # Линейное уравнение
        return [(x, a11 * x**2 + a12 * x + a13)]

    discriminant = b**2 - 4 * a * c

    if discriminant > 0:
        sqrt_d = math.sqrt(discriminant)
        x1 = (-b + sqrt_d) / (2 * a)
        x2 = (-b - sqrt_d) / (2 * a)
        return [(x1, a11 * x1**2 + a12 * x1 + a13), (x2, a11 * x2**2 + a12 * x2 + a13)]
    
    if discriminant == 0:
        x = -b / (2 * a)
        return [(x, a11 * x**2 + a12 * x + a13)]
    
    return []  # Нет реальных решений (дискриминант < 0)

def skew(x):
    n = len(x)
    
    mean_x = sum(x) / n
    
    variance = sum((xi - mean_x) ** 2 for xi in x) / n
    std_x = variance ** 0.5
    
    m3 = sum((xi - mean_x) ** 3 for xi in x) / n
    
    skewness = m3 / std_x**3
    
    return round(skewness, 2)


def kurtosis(x):
    n = len(x)
    
    mean_x = sum(x) / n
    
    variance = sum((xi - mean_x) ** 2 for xi in x) / n
    std_x = variance ** 0.5
    
    m4 = sum((xi - mean_x) ** 4 for xi in x) / n
    
    # Коэффициент эксцесса
    kurt = m4 / std_x**4 - 3
    
    return round(kurt, 2)
