from typing import List


def multiplication(a: float, vector: List[float]) -> List[float]:
    '''Тензорное умножение вектора на константу.
    :param a: Целочисленное значение(постоянная).
    :param vector: Вектор из целочисленных значений.
        :return: Список из компонент, получившихся в результате умножения.'''

    result: List[float] = [a * i for i in vector]
    return result


def addition(vectors: List[List], sign: bool = True) -> List[float]:
    """Производит сложение или вычитание некоторого количества векторов.
    :param vectors: Список векторов, над которыми будет производится операция. Векторы должны быть одинаковой длины.
    :param sign: Логическое значение, определяющее знак операции: для сложения - True, для вычетания - False. По умолчанию True.
        :return: Список из компонент, получившихся в результате сложения или вычитания."""

    prev_vector = 0

    for i, vector in enumerate(vectors):
        if vector is not True:
            raise Exception('Вектор не может быть пустым.')
        if i == 0:
            prev_vector = len(vector)
        if i > 0 and len(vector) != prev_vector:
            raise Exception('Векторы имеют разную размерность.')

    result: List[float] = []

    for i in range(len(vectors[0])):
        sum = 0

        for vector in vectors:
            if sign:
                sum += vector[i]
            else:
                sum -= vector[i]

        result.append(sum)

    return result
