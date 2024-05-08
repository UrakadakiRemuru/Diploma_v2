import math
from typing import List

from matplotlib import pyplot as plt

from calculations import initialize, lambda_eps_tensor_calculate, effective_stiffness_calculate_linear_maxwell_method, \
    lambda_sig_tensor_calculate, effective_compliance_calculate_linear_maxwell_method


def lm_stiffness_tensor_by_lame(lame_coefficients_list: List[List], n: int, volume: float, matrix_const: List[float], inhomo_size: List[float]):
    '''
    Меняем коэффы ламе и смотрим, что происходит
    :param lame_coefficients_list: Массив коэффициентов Ламе
    :param n: Количество неоднородностей
    :param volume: Репрезентативный элемент объема
    :return: Массив тензоров жесткости.
    '''

    C_eff_list = []

    for i in lame_coefficients_list:
        structure = initialize(matrix_const, [inhomo_size for _ in range(n)], [i for _ in range(n)],
                               ['spheroid' for _ in range(n)])
        lambda_tensors = lambda_eps_tensor_calculate(structure)
        res = effective_stiffness_calculate_linear_maxwell_method(structure, lambda_tensors, volume)
        C_eff_list.append(res)

    # for i in C_eff_list:
    #     print(i)

    return C_eff_list


def lm_stiffness_tensor_by_volume_fraction(volume: float, n: int, fi_list: List[float], matrix_const: List,
                                        inhomo_const: List):
    '''
    Зависимость коэффициентов тензора модулей упругости от объемной доли
    :param volume: Репрезентативный элемент объема
    :param n: Количество неоднородностей
    :param fi_list: Список объемных долей
    :param matrix_const: Коэффициенты Ламе матрицы
    :param inhomo_const: Коэффициенты Ламе неоднородности
    '''

    b_list = [3 / 4 * volume / n * fi / math.pi / 1 for fi in fi_list]
    C_eff_list_plot = []

    for b in b_list:
        structure = initialize(matrix_const, [[1, b] for _ in range(n)], [inhomo_const for _ in range(n)],
                               ['spheroid' for _ in range(n)])
        lambda_tensors = lambda_eps_tensor_calculate(structure)
        C_eff_list_plot.append(effective_stiffness_calculate_linear_maxwell_method(structure, lambda_tensors, volume))

    C_list = []
    FI_list = []

    for tensor in C_eff_list_plot:
        for i, component in enumerate(tensor):
            if i == 0:
                print('компоненты', component)
                C_list.append(component)
            elif i == 2:
                FI_list.append(component)
                print('доля', component)

    smth = zip(*C_list)
    num = 1
    for el in smth:
        plt.plot(FI_list, list(el), label=f'C_{num}')
        num += 1
    plt.title('Учет взаимодействия линеаризованным методом Максвелла')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    plt.xlabel('Объемная доля')
    plt.show()

def lm_compliance_tensor_by_volume_fraction(volume: float, n: int, fi_list: List[float], matrix_const: List,
                                        inhomo_const: List):
    '''
    Зависимость коэффициентов тензора модулей упругости от объемной доли
    :param volume: Репрезентативный элемент объема
    :param n: Количество неоднородностей
    :param fi_list: Список объемных долей
    :param matrix_const: Коэффициенты Ламе матрицы
    :param inhomo_const: Коэффициенты Ламе неоднородности
    '''

    b_list = [3 / 4 * volume / n * fi / math.pi / 1 for fi in fi_list]
    S_eff_list_plot = []

    for b in b_list:
        structure = initialize(matrix_const, [[1, b] for _ in range(n)], [inhomo_const for _ in range(n)],
                               ['spheroid' for _ in range(n)])
        lambda_tensors = lambda_sig_tensor_calculate(structure)
        S_eff_list_plot.append(effective_compliance_calculate_linear_maxwell_method(structure, lambda_tensors, volume))

    S_list = []
    FI_list = []

    for tensor in S_eff_list_plot:
        for i, component in enumerate(tensor):
            if i == 0:
                print('компоненты', component)
                S_list.append(component)
            elif i == 2:
                FI_list.append(component)
                print('доля', component)

    smth = zip(*S_list)
    num = 1
    for el in smth:
        plt.plot(FI_list, list(el), label=f'S_{num}')
        num += 1
    plt.title('Учет взаимодействия линеаризованным методом Максвелла')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    plt.xlabel('Объемная доля')
    plt.show()