from math import e, pow, log

def dot_product(v1: list[float], v2: list[float]) -> float:

    if len(v1) != len(v2):
        raise ValueError('Vectors must be the same length')

    multipliedNumbers = [value * v2[i] for i, value in enumerate(v1)]

    return sum(multipliedNumbers)

def exp(x: float):
    return pow(e, x)

def ln(x: float):
    return log(x, e)

def tensorMultiply(m1: list[list[float]], m2: list[list[float]]):
    return [
        []
    ]