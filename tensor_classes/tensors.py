import math
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
        self._manual_components = False


    @property
    def components(self) -> Annotated[List[float], 6]:
        '''Возвращает список компонент разложения трансверсально-изотропного тензора четвертого ранга в тензорном базисе.'''
        if not self._manual_components:
            try:
                self._components = addition([multiplication(self.constants[0], self.E1),
                                     multiplication(self.constants[1], self.E2)])
            except AttributeError:
                pass
        return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value
        self._manual_components = False

    @components.deleter
    def components(self):
        del self._components
        self._manual_components = False

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
        self.K = constants[0] + 2 / 3 * constants[1]
        self.constants = constants

class ResultTransverselyIsotropicTensor(TransverselyIsotropicTensor):
    def __init__(self, components: Annotated[List[float], 6]):
        super().__init__([0, 0])
        self._components = components
        self._manual_components = False

    @property
    def components(self) -> Annotated[List[float], 6]:
        return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value
        self._manual_components = True

    @components.deleter
    def components(self):
        del self._components
        self._manual_components = False

class ResultTransverselyIsotropicTensor(TransverselyIsotropicTensor):
    def __init__(self, components: Annotated[List[float], 6]):
        super().__init__([0, 0])
        self._components = components
        self._manual_components = False

    @property
    def components(self) -> Annotated[List[float], 6]:
        return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value
        self._manual_components = True

    @components.deleter
    def components(self):
        del self._components
        self._manual_components = False

class HillsTensor(TransverselyIsotropicTensor):
    def __init__(self, constants: Annotated[List[float], 2], size: Annotated[List[float], 3 | 2 | 1]):
        super().__init__(constants)
        self.l, self.m = self.constants[0], self.constants[1]
        self.__size: Annotated[List[float], 3 | 2 | 1] = size
        if len(self.__size) == 1:
            self.gamma: float = 1
        elif len(self.__size) == 2:
            self.gamma: float = self.__size[1] / self.__size[0]
        self.k: float = (self.l + self.m) / (self.l + 2 * self.m)
        self._components = []
        self._manual_components = False
        if self.gamma == 1:
            self.g = 0
            self.f0 = 0
            self.f1 = 0
        else:
            self.g = 1 / self.gamma / math.sqrt(1 - self.gamma ** 2) * math.atan(
                math.sqrt(1 - self.gamma ** 2) / self.gamma)
            self.f0 = (1 - self.g) / 2 / (1 - self.gamma ** 2)
            self.f1 = 1 / 4 / (1 - self.gamma ** 2) ** 2 * (
                    (2 + self.gamma ** 2) * self.g - 3 * self.gamma ** 2)


    @property
    def components(self) -> Annotated[List[float], 6]:
        if not self._manual_components:
            if not self.__size:
                self._components = [0 for _ in range(6)]
            if len(self.__size) == 1:
                try:
                    self._components = addition([multiplication((1 - self.k) / 3 / self.m, self.E1),
                                                 multiplication((5 - 2 * self.k) / 15 / self.m, self.E2)])
                except AttributeError:
                    pass
            elif len(self.__size) == 2:
                self._components = [
                    1 / 2 / self.m * ((1 - self.k) * self.f0 + self.k * self.f1),
                    1 / 2 / self.m * ((2 - self.k) * self.f0 + self.k * self.f1),
                    - self.k / self.m * self.f1,
                    - self.k / self.m * self.f1,
                    1 / self.m * (1 - self.f0 - 4 * self.k * self.f1),
                    1 / self.m * ((1 - self.k) * (1 - 2 * self.f0) + 2 * self.k * self.f1)
                ]
        return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value
        self._manual_components = True

    @components.deleter
    def components(self):
        del self._components
        self._manual_components = False


class ComplianceTensor(TransverselyIsotropicTensor):

    def __init__(self, constants: Annotated[List[float], 2]):
        super(ComplianceTensor, self).__init__(constants)
        self._components = []
        self.K = constants[0] + 2 / 3 * constants[1]
        self.mu = constants[1]
        self._manual_components = False

    @property
    def components(self) -> Annotated[List[float], 6]:
        if not self._manual_components:
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

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value
        self._manual_components = True

    @components.deleter
    def components(self):
        del self._components
        self._manual_components = False

class DualHillsTensor(TransverselyIsotropicTensor):
    def __init__(self, constants: Annotated[List[float], 2], size: Annotated[List[float], 3 | 2 | 1]):
        super().__init__(constants)
        self.lambda_0, self.mu_0 = self.constants[0], self.constants[1]
        self.__size: Annotated[List[float], 3 | 2 | 1] = size
        if len(self.__size) == 1:
            self.gamma: float = 1
        elif len(self.__size) == 2:
            self.gamma: float = self.__size[1] / self.__size[0]
        self.kappa_0: float = (self.lambda_0 + self.mu_0) / (self.lambda_0 + 2 * self.mu_0)
        self._components = []
        self._manual_components = False
        if self.gamma == 1:
            self.g = 0
            self.f0 = 0
            self.f1 = 0
        else:
            self.g = 1 / self.gamma / math.sqrt(1 - self.gamma ** 2) * math.atan(
                math.sqrt(1 - self.gamma ** 2) / self.gamma)
            self.f0 = (1 - self.g) / 2 / (1 - self.gamma ** 2)
            self.f1 = 1 / 4 / (1 - self.gamma ** 2) ** 2 * (
                    (2 + self.gamma ** 2) * self.g - 3 * self.gamma ** 2)

    @property
    def components(self) -> Annotated[List[float], 6]:
        if not self._manual_components:
            if len(self.__size) == 1:
                try:
                    self._components = addition([multiplication(- 4 * self.mu_0 * (1 - 4 * self.kappa_0) / 3, self.E1),
                                                 multiplication(2 * self.mu_0 * (5 + 2 * self.kappa_0) / 15, self.E2)])
                except AttributeError:
                    pass
            elif len(self.__size) == 2:
                if self.gamma == 1:
                    self._components = []
                elif self.gamma < 0.15:
                    self._components = [
                        self.mu_0 * (4 * self.kappa_0 - 1 - pi * (7 * self.kappa_0 - 2) * self.gamma / 4),
                        2 * self.mu_0 * (1 - pi * (4 -  self.kappa_0) * self.gamma / 8),
                        pi * self.mu_0 * (3 * self.kappa_0 - 1) * self.gamma / 2,
                        pi * self.mu_0 * (3 * self.kappa_0 - 1) * self.gamma / 2,
                        pi * self.mu_0 * (2 * self.kappa_0 + 1) * self.gamma,
                        pi * self.mu_0 * self.kappa_0 * self.gamma
                    ]
                else:
                    self._components = [
                        (4 * self.kappa_0 - 1 - 2 * (3 * self.kappa_0 - 1) * self.f0 - 2 * self.f1),
                        2 * (1 - (2 - self.kappa_0) * self.f0 - self.f1),
                        2 * ((2 * self.kappa_0 - 1) * self.f0 + 2 * self.f1),
                        2 * ((2 * self.kappa_0 - 1) * self.f0 + 2 * self.kappa_0 * self.f1),
                        4 * (self.f0 + 4 * self.f1),
                        8 * (self.kappa_0 * self.f0 - self.f1)
                    ]
        return self._components

    @components.setter
    def components(self, value: Annotated[List[float], 6]):
        self._components = value
        self._manual_components = True

    @components.deleter
    def components(self):
        del self._components
        self._manual_components = False
