from math import pi
from typing import List, Annotated
from operations.operations import multiplication, addition


class TransverselyIsotropicTensor:
    def __init__(self, constants: Annotated[List[float], 2]):
        '''Представление трансверсально-изотропного тензора в тензорном базисе тензоров четвертого ранга.
        :param constants: Массив, состоящий из двух целочисленных значений - константы в изотропном представлении тензора.
        '''
        __useless: Annotated[List[float], 6] = [1 / 2, 1, 0, 0, 2, 1]
        self.constants = constants
        self.E1: Annotated[List[float], 6] = [1 / 3 if i not in [1, 4] else 0 for i in range(6)]
        self.E2: Annotated[List[float], 6] = [__useless[i] - self.E1[i] for i in range(6)]

    def components(self) -> Annotated[List[float], 6]:
        '''Возвращает список компонент разложения трансверсально-изотропного тензора четвертого ранга в тензорном базисе.'''

        result: List = addition([multiplication(self.constants[0], self.E1),
                                 multiplication(self.constants[1], self.E2)])
        return result

    def inverse(self) -> Annotated[List[float], 6]:
        '''Возвращает список компонент разложения обратного трансверсально-изотропного тензора четвертого ранга в тензорном базисе.'''

        a, b, c, d, e, f = self.components()
        delta = 2 * (a * f - c * d)
        return [f / 2 / delta, 1 / b, - c / delta, - d / delta, 4 / e, 2 * a / delta]


class ElasticStiffnessTensor(TransverselyIsotropicTensor):
    def __init__(self, constants: Annotated[List[float], 2]):
        '''
        Представление тензора жесткости в тензорном базисе тензоров четвертого ранга.
        :param constants: Массив, состоящий из двух целочисленных значений - коэффициенты Ламе [λ, μ].
        '''
        super(ElasticStiffnessTensor, self).__init__(constants)
        self.E2 = multiplication(2, addition([self.E2, self.E1]))
        self.E1 = multiplication(3, self.E1)


class ResultTransverselyIsotropicTensor(TransverselyIsotropicTensor):
    def __init__(self, components: Annotated[List[float], 6]):
        super().__init__([0, 0])
        self.components = components
        del self.E1
        del self.E2
        del self.constants

    def components(self) -> Annotated[List[float], 6]:
        return self.components
