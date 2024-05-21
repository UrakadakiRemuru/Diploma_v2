import math
from typing import List, Annotated
from math import pi


class IsotropicTensor:
    '''Представление изотропного тензора.'''

    def __init__(self, params: Annotated[List[float], 2]):
        '''
        :param params: Список из двух целочисленных значений - параметров Ламе([λ, μ]).
        '''
        self.components = params
        self.constants = params

    def inverse(self):
        '''Возвращает список компонент разложения обратного трансверсально-изотропного тензора четвертого ранга в тензорном базисе.'''

        a, b = self.components

        if a == b == 0:
            return IsotropicTensor([0 for _ in range(2)])
        return IsotropicTensor([1 / a, 1 / b])


class ElasticStiffnessTensor(IsotropicTensor):
    def __init__(self, params: Annotated[List[float], 2]):
        l, m = params
        super().__init__([2 * l + m, 14 / 3 * m])

class ComplianceTensor(IsotropicTensor):
    def __init__(self, params: Annotated[List[float], 2]):
        l, m = params
        super().__init__([-(30 * l + 23 * m) / 10 / m / (3 * l + 2 * m), 7 / 10 / m])

class HillsTensor(IsotropicTensor):
    def __init__(self, params: Annotated[List[float], 2], size: Annotated[List[float], 3 | 2 | 1]):
        l, m = params
        size: Annotated[List[float], 3 | 2 | 1] = size
        k: float = (l + m) / (l + 2 * m)
        if len(size) == 2:
            if size[0] == 0:
                p1 = p2 = p3 = p4 = p5 = p6 = 0
            else:
                gamma: float = size[1] / size[0]
                g = 1 / gamma / math.sqrt(1 - gamma ** 2) * math.atan(
                    math.sqrt(1 - gamma ** 2) / gamma)
                f0 = (1 - g) / 2 / (1 - gamma ** 2)
                f1 = 1 / 4 / (1 - gamma ** 2) ** 2 * (
                        (2 + gamma ** 2) * g - 3 * gamma ** 2)
                p1 = 1 / 2 / m * ((1 - k) * f0 + k * f1)
                p2 = 1 / 2 / m * ((2 - k) * f0 + k * f1)
                p3 = p4 = - k / m * f1
                p5 = 1 / m * (1 - f0 - 4 * k * f1)
                p6 = 1 / m * ((1 - k) * (1 - 2 * f0) + 2 * k * f1)
        elif len(size) == 1:
            p1 = (5 - 4 * k) / 30 / m
            p2 = (10 - 4 * k) / 30 / m
            p3 = p4 = - k / m * 15
            p5 = (10 - 4 * k) / 15 / m
            p6 = (5 - 3 * k) / 15 / m
        super().__init__([
            1 / 15 * (2 * p1 + 6 * p2 - 4 * p3 + 6 * p5 + 2 * p6),
            1 / 15 * (p1 - 2 * p2 + 8 * p3 - 2 * p5 + p6)
        ])

