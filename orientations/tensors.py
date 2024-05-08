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
        g: float = size[1] / size[0]
        k: float = (l + m) / (l + 2 * m)
        super().__init__([1 / 120 / m * (40 - 40 * k - pi * (15 + 37 / 2 * k) * g), 1 / 120 / m * (28 - 16 * k + pi * (3 + k) * g)])

