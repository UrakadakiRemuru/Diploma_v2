from typing import Annotated, List, Tuple, Union

from tensor_classes.tensors import (TransverselyIsotropicTensor,
                                    ResultTransverselyIsotropicTensor,
                                    ElasticStiffnessTensor,
                                    HillsTensor, ComplianceTensor)
from operations.tensor_operations import addition, double_dot_product, multiplication
from Inhomogeneities.Inhomogeneities import inhomogeneity


def initialize(
        matrix_const: Annotated[List[float], 2],
        inhomogeneity_size_list: List[Annotated[List[float], 3 | 2 | 1]],
        inhomogeneity_const_list: List[Annotated[List[float], 2]],
        inhomogenetiy_type_list: List[str]
) -> List[Union[ElasticStiffnessTensor, List[inhomogeneity]]]:
    '''
    Инициализация тензорa жесткости матрицы и неоднородностей.
    :param matrix_const: Массив коэффициентов Ламе матрицы [λ, μ].
    :param inhomogeneity_size_list: Массив из массивов размерностей неоднородности.
    :param inhomogeneity_const_list: Массив из массивов коэффициентов Ламе.
    :param inhomogenetiy_type_list: Массив из типов неоднородностей.
        :return: Возвращает массив из тензора жеткости и податливости матрицы и неоднородностей.
    '''

    return [
        ElasticStiffnessTensor(matrix_const),
        ComplianceTensor(matrix_const),
        [inhomogeneity(a, b, c) for a, b, c in
         zip(inhomogeneity_size_list, inhomogeneity_const_list, inhomogenetiy_type_list)]
    ]


def lambda_eps_tensor_calculate(structure: List[Union[ElasticStiffnessTensor, List[inhomogeneity]]]) -> List[
    ResultTransverselyIsotropicTensor]:
    '''
    Вычисление тензора Λ_e.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
        :return: Массив тензоров Λ_e.
    '''

    I = TransverselyIsotropicTensor([1, 1])
    C_0 = structure[0]
    inhomos = structure[2]

    result = []
    for inhomo in inhomos:
        step_1 = addition([inhomo.stiffness_tensor, C_0], False)
        step_2 = double_dot_product([inhomo.hills_tensor, step_1])
        step_3 = addition([I, step_2])
        result.append(step_3.inverse())

    return result


def lambda_sig_tensor_calculate(structure: List[Union[ComplianceTensor, List[inhomogeneity]]]) -> List[
    ResultTransverselyIsotropicTensor]:
    '''
    Вычисление тензора Λ_g.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
        :return: Массив тензоров Λ_g.
    '''

    I = TransverselyIsotropicTensor([1, 1])
    S_0 = structure[1]
    inhomos = structure[2]

    result = []
    for inhomo in inhomos:
        step_1 = addition([inhomo.compliance_tensor, S_0], False)
        step_2 = double_dot_product([inhomo.dual_hills_tensor, step_1])
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
        structure: List[Union[ElasticStiffnessTensor, List[inhomogeneity]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ_e.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[2]
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

def effective_compliance_calculate(
        structure: List[Union[ComplianceTensor, List[inhomogeneity]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор податливости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ_g.
        :return: Эффективный тензор податливости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    S_0 = structure[1]
    inhomos = structure[2]
    smth: list = []
    fi: float = 0
    for inhomo, lam in zip(inhomos, lambda_list):
        dev_v = inhomo.volume / volume
        fi += dev_v
        step_1 = addition([inhomo.compliance_tensor, S_0], False)
        step_2 = multiplication(dev_v, step_1)
        step_3 = double_dot_product([step_2, lam])
        smth.append(step_3)
        l, m = inhomo.stiffness_tensor.constants

    S_eff = addition(smth + [S_0]).components
    return [S_eff, is_elastic_modules_exist(S_eff), [l / S_0.constants[0], m / S_0.constants[1]], fi]

def effective_stiffness_calculate_maxwell_method(
        structure: List[Union[ElasticStiffnessTensor, List[inhomogeneity]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[2]
    smth: list = []
    fi = 0

    for inhomo, lam in zip(inhomos, lambda_list):
        dev_v = inhomo.volume / volume
        fi += dev_v
        C_1 = inhomo.stiffness_tensor
        lambd = lam
        l, m = inhomo.stiffness_tensor.constants

    if fi == 0:
        C_eff = C_0.components
    else:
        step_1 = addition([C_1, C_0], False)
        step_2 = double_dot_product([step_1, lambd]).inverse()
        step_3 = multiplication(1 - fi, step_2)
        step_4 = step_1.inverse()
        step_5 = multiplication(fi, step_4)
        step_6 = addition([step_3, step_5]).inverse()
        step_7 = multiplication(fi, step_6)
        C_eff = addition([C_0, step_7]).components

    return [C_eff, is_elastic_modules_exist(C_eff), [l / C_0.constants[0], m / C_0.constants[1]], fi]


def effective_compliance_calculate_maxwell_method(
        structure: List[Union[ComplianceTensor, List[inhomogeneity]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор податливости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ_g.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    S_0 = structure[1]
    inhomos = structure[2]
    smth: list = []
    fi = 0

    for inhomo, lam in zip(inhomos, lambda_list):
        dev_v = inhomo.volume / volume
        fi += dev_v
        S_1 = inhomo.compliance_tensor
        lambd = lam
        l, m = inhomo.compliance_tensor.constants

    if fi == 0:
        S_eff = S_0.components
    else:
        step_1 = addition([S_1, S_0], False)
        step_2 = double_dot_product([step_1, lambd]).inverse()
        step_3 = multiplication(1 - fi, step_2)
        step_4 = step_1.inverse()
        step_5 = multiplication(fi, step_4)
        step_6 = addition([step_3, step_5]).inverse()
        step_7 = multiplication(fi, step_6)
        S_eff = addition([S_0, step_7]).components

    return [S_eff, is_elastic_modules_exist(S_eff), [l / S_0.constants[0], m / S_0.constants[1]], fi]


def effective_stiffness_calculate_linear_maxwell_method(
        structure: List[Union[ElasticStiffnessTensor, List[inhomogeneity]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[2]
    fi = 0
    N_list = []

    for inhomo, lam in zip(inhomos, lambda_list):
        dev_v = inhomo.volume / volume
        fi += dev_v
        C_1 = inhomo.stiffness_tensor
        P = inhomo.hills_tensor
        dC = addition([C_1, C_0], False)
        N = double_dot_product([dC, lam])
        N_list.append(multiplication(dev_v, N))
        l, m = inhomo.stiffness_tensor.constants

    if fi == 0:
        C_eff = C_0.components
    else:
        N_sum = addition(N_list)
        step_1 = double_dot_product([P, N_sum])
        step_2 = double_dot_product([N_sum, step_1])
        C_eff = addition([C_0, N_sum, step_2]).components

    return [C_eff, is_elastic_modules_exist(C_eff), [l / C_0.constants[0], m / C_0.constants[1]], fi]

def effective_compliance_calculate_linear_maxwell_method(
        structure: List[Union[ComplianceTensor, List[inhomogeneity]]],
        lambda_list: List[ResultTransverselyIsotropicTensor],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор податливости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ_g.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    S_0 = structure[1]
    inhomos = structure[2]
    fi = 0
    H_list = []

    for inhomo, lam in zip(inhomos, lambda_list):
        dev_v = inhomo.volume / volume
        fi += dev_v
        S_1 = inhomo.compliance_tensor
        Q = inhomo.dual_hills_tensor
        dS = addition([S_1, S_0], False)
        H = double_dot_product([dS, lam])
        H_list.append(multiplication(dev_v, H))
        l, m = inhomo.compliance_tensor.constants

    if fi == 0:
        S_eff = S_0.components
    else:
        H_sum = addition(H_list)
        step_1 = double_dot_product([Q, H_sum])
        step_2 = double_dot_product([H_sum, step_1])
        S_eff = addition([S_0, H_sum, step_2]).components

    return [S_eff, is_elastic_modules_exist(S_eff), [l / S_0.constants[0], m / S_0.constants[1]], fi]

def effective_stiffness_calculate_kanaun_levin_method(
        structure: List[Union[ElasticStiffnessTensor, List[ElasticStiffnessTensor]]],
        volume: float
) -> List[Union[ResultTransverselyIsotropicTensor, str, Annotated[List[float], 2], float]]:
    '''
    Находит эффективный тензор жесткости, соотношение коэффициентов Ламе для матрицы и неоднородности, и объемную долю.
    :param structure: Массив из тензора жеткости матрицы и неоднородностей.
    :param lambda_list: Массив тензоров Λ.
        :return: Эффективный тензор жесткости, соотношение коэффициентов Ламе для неоднородности и матрицы, и объемную долю.
    '''

    C_0 = structure[0]
    inhomos = structure[2]
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
        pass
    C_eff = addition([C_0, step_4]).components
    return [C_eff, is_elastic_modules_exist(C_eff), [l / C_0.constants[0], m / C_0.constants[1]], fi]

