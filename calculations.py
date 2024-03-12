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

def is_elastic_modules_exist(c_eff_components: list[float]) -> str:
    '''
    Проверка существования полученного материала.
    :param c_eff_components: Компоненты тензора эффективной упругости.
    :return: Существует материал или нет.
    '''
    m = c_eff_components[1] / 2
    l = c_eff_components[0] - m

    if m <= 0 and l + 2 * m <= 0:
        return f"Такой материал не может существовать, так как μ = {m} <= 0 и λ + 2μ = {l + 2 * m} <= 0."

    if m <= 0:
        return f"Такой материал не может существовать, так как μ = {m} <= 0."

    if l + 2 * m <= 0:
        return f"Такой материал не может существовать, так как λ + 2μ = {l + 2 * m} <= 0."

    return "Такой материал может существовать."

def effective_stiffness_calculate(
        structure: List[Union[ElasticStiffnessTensor, List[ElasticStiffnessTensor]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str,  Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[1]
    smth: list = []
    fi: float = 0
    for inhomo, lam in zip(inhomos, lambda_list):
        dev_v = inhomo.volume / volume
        fi += dev_v
        step_1 = addition([inhomo.stiffness_tensor, C_0], False)
        step_2 = multiplication(dev_v, step_1)
        step_3 = double_dot_product([step_2, lam])
        smth.append(step_3)
        l, m = inhomo.stiffness_tensor.constants

    C_eff = addition(smth + [C_0]).components
    return [C_eff, is_elastic_modules_exist(C_eff), [l / C_0.constants[0], m / C_0.constants[1]], fi]


def effective_stiffness_calculate_maxwell_method(
        structure: List[Union[ElasticStiffnessTensor, List[ElasticStiffnessTensor]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str,  Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[1]
    smth: list = []
    fi = 0
    b = 3 / 4 * volume

    for inhomo, lam in zip(inhomos, lambda_list):
        dev_v = inhomo.volume / volume
        fi += dev_v
        P = inhomo.hills_tensor
        step_1 = addition([inhomo.stiffness_tensor, C_0], False).inverse()
        step_2 = addition([step_1, P]).inverse()
        step_3 = multiplication(dev_v, step_2)
        step_4 = double_dot_product([step_2, lam])
        smth.append(step_4)
        l, m = inhomo.stiffness_tensor.constants


    if fi == 0:
        C_eff = C_0.components
    else:
        N = addition(smth).inverse()
        inv_add = addition([N, P], False).inverse()
        C_eff = addition([C_0, inv_add]).components

    return [C_eff, is_elastic_modules_exist(C_eff), [l / C_0.constants[0], m / C_0.constants[1]], fi]

def effective_stiffness_calculate_kanaun_levin_method(
        structure: List[Union[ElasticStiffnessTensor, List[ElasticStiffnessTensor]]],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str,  Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[1]
    fi = 0
    for inhomo in inhomos:
        dev_v = inhomo.volume / volume
        fi += dev_v
        C_1 = inhomo.stiffness_tensor
        P = inhomo.hills_tensor
        l, m = inhomo.stiffness_tensor.constants
    print('Доля:', fi)
    print('Тензор Хилла:', P)
    print('Неоднородность', C_1)
    print('Матрица', C_0)
    step_1 = addition([C_1, C_0], False).inverse()
    step_2 = multiplication(1 - fi, P)
    step_3 = addition([step_1, step_2]).inverse()
    step_4 = multiplication(fi, step_3)

    if fi == 0:
        C_eff = C_0.components
    else:
        C_eff = addition([C_0, step_4]).components

    return [C_eff, is_elastic_modules_exist(C_eff), [l / C_0.constants[0], m / C_0.constants[1]], fi]