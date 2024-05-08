from math import pi
from typing import List, Annotated, Dict
from operations.operations import multiplication, addition


class TransverselyIsotropicTensor:
    def __init__(self, constants: Annotated[List[float], 2 | 6]):
        '''Представление трансверсально-изотропного тензора в тензорном базисе тензоров четвертого ранга.
        :param constants: Массив, состоящий из двух целочисленных значений - константы в изотропном представлении тензора.
        '''
        __useless: Annotated[List[float], 6] = [1 / 2, 1, 0, 0, 2, 1]
        if not constants:
            raise Exception('Пустой массив.')
        if len(constants) == 2:
            self.constants = constants
        else:
            self._components: Annotated[List[float], 6] = constants
        self.E1: Annotated[List[float], 6] = [1 / 3 if i not in [1, 4] else 0 for i in range(6)]
        self.E2: Annotated[List[float], 6] = [__useless[i] - self.E1[i] for i in range(6)]


    @property
    def components(self) -> Annotated[List[float], 6]:
        '''Возвращает список компонент разложения трансверсально-изотропного тензора четвертого ранга в тензорном базисе.'''
        try:
            self._components = addition([multiplication(self.constants[0], self.E1),
                                 multiplication(self.constants[1], self.E2)])
        except AttributeError:
            pass
        return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value

    @components.deleter
    def components(self):
        del self._components

    def inverse(self):
        '''Возвращает список компонент разложения обратного трансверсально-изотропного тензора четвертого ранга в тензорном базисе.'''

        a, b, c, d, e, f = self.components
        delta = 2 * (a * f - c * d)
        if delta == 0:
            return TransverselyIsotropicTensor([0 for _ in range(6)])
        return TransverselyIsotropicTensor([f / 2 / delta, 1 / b, - c / delta, - d / delta, 4 / e, 2 * a / delta])


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
        self._components = components

    @property
    def components(self) -> Annotated[List[float], 6]:
        return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value

    @components.deleter
    def components(self):
        del self._components

class HillsTensor(TransverselyIsotropicTensor):
    def __init__(self, constants: Annotated[List[float], 2], size: Annotated[List[float], 3 | 2 | 1]):
        super().__init__(constants)
        self.__lambda_0, self.__mu_0 = self.constants[0], self.constants[1]
        self.__size: Annotated[List[float], 3 | 2 | 1] = size
        self.__gamma: float = self.__size[1] / self.__size[0]
        self.__kappa_0: float = (self.__lambda_0 + self.__mu_0) / (self.__lambda_0 + 2 * self.__mu_0)
        self._components = []

    @property
    def components(self) -> Annotated[List[float], 6]:
        if len(self.__size) == 1:
            try:
                self._components = addition([multiplication((1 - self.__kappa_0) / 3 / self.__mu_0, self.E1),
                                             multiplication((5 - 2 * self.__kappa_0) / 15 / self.__mu_0, self.E2)])
            except AttributeError:
                pass
            return self._components
        elif len(self.__size) == 2:
            self._components = [
                pi * (2 - self.__kappa_0) * self.__gamma / 16 / self.__mu_0,
                pi * (4 - self.__kappa_0) * self.__gamma / 16 / self.__mu_0,
                pi * (- self.__kappa_0) * self.__gamma / 8 / self.__mu_0,
                pi * (- self.__kappa_0) * self.__gamma / 8 / self.__mu_0,
                (1 - pi * (1 + self.__kappa_0) * self.__gamma / 4) / 4 / self.__mu_0,
                (1 - self.__kappa_0 - pi * (2 - self.__kappa_0) * self.__gamma / 4) / self.__mu_0
            ]
            return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value

    @components.deleter
    def components(self):
        del self._components


class ComplianceTensor(TransverselyIsotropicTensor):

    def __init__(self, constants: Annotated[List[float], 2]):
        super(ComplianceTensor, self).__init__(constants)
        self._components = []

    @property
    def components(self) -> Annotated[List[float], 6]:
        l, m = self.constants
        self._components = [
            (l + 2 * m) / 4 / m / (3 * l + 2 * m),
            1 / 2 / m,
            - l / 2 / m / (3 * l + 2 * m),
            - l / 2 / m / (3 * l + 2 * m),
            1 / m,
            2 * (l + m) / 2 / m / (3 * l + 2 * m)
        ]
        return self._components

class DualHillsTensor(TransverselyIsotropicTensor):
    def __init__(self, constants: Annotated[List[float], 2], size: Annotated[List[float], 3 | 2 | 1]):
        super().__init__(constants)
        self.__lambda_0, self.__mu_0 = self.constants[0], self.constants[1]
        self.__size: Annotated[List[float], 3 | 2 | 1] = size
        self.__gamma: float = self.__size[1] / self.__size[0]
        self.__kappa_0: float = (self.__lambda_0 + self.__mu_0) / (self.__lambda_0 + 2 * self.__mu_0)
        self._components = []

    @property
    def components(self) -> Annotated[List[float], 6]:
        if len(self.__size) == 1:
            try:
                self._components = addition([multiplication(- 4 * self.__mu_0 * (1 - 4 * self.__kappa_0) / 3, self.E1),
                                             multiplication(2 * self.__mu_0 * (5 + 2 * self.__kappa_0) / 15, self.E2)])
            except AttributeError:
                pass
            return self._components
        elif len(self.__size) == 2:
            self._components = [
                self.__mu_0 * (4 * self.__kappa_0 - 1 - pi * (7 * self.__kappa_0 - 2) * self.__gamma / 4),
                2 * self.__mu_0 * (1 - pi * (4 -  self.__kappa_0) * self.__gamma / 8),
                pi * self.__mu_0 * (3 * self.__kappa_0 - 1) * self.__gamma / 2,
                pi * self.__mu_0 * (3 * self.__kappa_0 - 1) * self.__gamma / 2,
                pi * self.__mu_0 * (2 * self.__kappa_0 + 1) * self.__gamma,
                pi * self.__mu_0 * self.__kappa_0 * self.__gamma
            ]
            return self._components
