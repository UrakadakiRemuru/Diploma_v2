from typing import List, Annotated
from math import pi

from tensor_classes.tensors import HillsTensor, ElasticStiffnessTensor


class inhomogeneity:
    '''Представление неоднородности.'''
    def __init__(self, size: Annotated[List[float], 3 | 2 | 1], const: List[float], inhomo_type: str = ''):
        '''
        Создает неоднородность с заданными размерами, физическими свойствами и типом.
        :param size: Массив размерностей неоднородности в трехмерном пространстве [a1, a2, a3]. В зависимости от типа неоднородности этот массив имеет различную размерность. 'sphere' - 1 компонента [a], 'spheroid' - 2 компоненты [a1 = a2 = a, a3], 'ellipsoid' - 3 компоненты [a1, a2, a3].
        :param const: Физические константы Ламе [λ, μ], позволяющие задать тензор жесткости.
        :param inhomo_type: Один из трех типов, определяющий геометрическую форму неоднородности. 'spehre' - cфера, 'spheroid' - сфероид, 'ellipsoid' - эллипсоид. '''

        self.__type_list: List[str] = ['sphere', 'spheroid', 'ellipsoid']

        if const[1] <= 0:
            raise Exception(f"Такая неоднородность не может существовать, так как μ = {const[1]} <= 0! Необходимо, чтобы выполнялось условие μ > 0.")
        if const[0] + 2 * const[1] <= 0:
            raise Exception(f"Такая неоднородность не может существовать, так как λ + 2μ = {const[0] + 2 * const[1]} <= 0! Необходимо, чтобы выполнялось условие λ + 2μ > 0.")
        if inhomo_type not in self.__type_list:
            raise Exception(f'Тип должен совпадать с одним из указанного списка из {self.__type_list} .')

        if self.__type_list[0] == inhomo_type and len(size) != 1:
            raise Exception('Массив размерностей должен содержать одну компоненту.')
        elif self.__type_list[1] == inhomo_type and len(size) != 2:
            raise Exception('Массив размерностей должен содержать две компоненты.')
        elif self.__type_list[2] == inhomo_type and len(size) != 3:
            raise Exception('Массив размерностей должен содержать три компоненты.')

        self.size: Annotated[List[float], 3 | 2 | 1] = size
        self.stiffness_tensor = ElasticStiffnessTensor(const)
        self.inhomo_type: str = inhomo_type
        self.hills_tensor = HillsTensor(const, self.size)

    @property
    def volume(self) -> float:
        '''Возвращает объем неоднородности.'''

        if self.inhomo_type == self.__type_list[0]:
            return 4 / 3 * pi * self.size[0] ** 3
        elif self.inhomo_type == self.__type_list[1]:
            return 4 / 3 * pi * self.size[0] ** 2 * self.size[1]
        elif self.inhomo_type == self.__type_list[2]:
            return 4 / 3 * pi * self.size[0] * self.size[1] * self.size[2]

