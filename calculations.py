from typing import Annotated, List, Tuple, Union

from tensor_classes.tensors import (TransverselyIsotropicTensor,
                                    ResultTransverselyIsotropicTensor,
                                    ElasticStiffnessTensor,
                                    HillsTensor)
from operations.tensor_operations import addition, double_dot_product, multiplication
from Inhomogeneities.Inhomogeneities import inhomogeneity



def initialize(
        matrix_const: Annotated[List[float], 2],
        inhomogeneity_size_list: List[Annotated[List[float], 3 | 2 | 1]],
        inhomogeneity_const_list: List[Annotated[List[float], 2]],
        inhomogenetiy_type_list: List[str]
) -> List[Union[ElasticStiffnessTensor, List[ElasticStiffnessTensor]]]:
    '''
    Инициализация тензорa жесткости матрицы и неоднородностей.
    :param matrix_const: Массив коэффициентов Ламе матрицы [λ, μ].
    :param inhomogeneity_size_list: Массив из массивов размерностей неоднородности.
    :param inhomogeneity_const_list: Массив из массивов коэффициентов Ламе.
    :param inhomogenetiy_type_list: Массив из типов неоднородностей.
        :return: Возвращает массив из тензора жеткости матрицы и неоднородностей.
    '''

    return [
        ElasticStiffnessTensor(matrix_const),
        [inhomogeneity(a, b, c) for a, b, c in
         zip(inhomogeneity_size_list, inhomogeneity_const_list, inhomogenetiy_type_list)]
    ]


def lambda_tensor_calculate(structure: List[Union[ElasticStiffnessTensor, List[ElasticStiffnessTensor]]]) -> List[
    ResultTransverselyIsotropicTensor]:
    '''
    Вычисление тензора Λ.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
        :return: Массив тензоров Λ.
    '''

    I = TransverselyIsotropicTensor([1, 1])
    C_0 = structure[0]
    inhomos = structure[1]

    result = []
    for inhomo in inhomos:
        step_1 = addition([inhomo.stiffness_tensor, C_0], False)
        step_2 = double_dot_product([inhomo.hills_tensor, step_1])
        step_3 = addition([I, step_2])
        result.append(step_3.inverse())

    return result


def effective_stiffness_calculate(
        structure: List[Union[ElasticStiffnessTensor, List[ElasticStiffnessTensor]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[1]
    smth: list = []
    fi: float = 0
    for inhomo, lam in zip(inhomos, lambda_list):
        fi += inhomo.volume / volume
        step_1 = addition([inhomo.stiffness_tensor, C_0], False)
        step_2 = multiplication(fi, step_1)
        step_3 = double_dot_product([step_2, lam])
        smth.append(step_3)
        l, m = inhomo.stiffness_tensor.constants
    return [addition(smth + [C_0]).components, [l / C_0.constants[0] , m / C_0.constants[1]], fi]


