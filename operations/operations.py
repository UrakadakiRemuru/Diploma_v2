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
    prev_vector: float = 0

    for i, vector in enumerate(vectors):
        if not vector:
            raise Exception('Вектор не может быть пустым.')
        if i == 0:
            prev_vector = len(vector)
        if i > 0 and len(vector) != prev_vector:
            raise Exception('Векторы имеют разную размерность.')

    result: List[float] = []

    for i in range(len(vectors[0])):
        sum: float = 0

        for j, vector in enumerate(vectors):
            if sign:
                sum += vector[i]
            else:
                if j == 0:
                    sum = vector[i]
                else:
                    sum -= vector[i]

        result.append(sum)

    return result

def E_nu_to_lam_mu(E: float, nu: float) -> List[float]:
    return [nu * E / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)]