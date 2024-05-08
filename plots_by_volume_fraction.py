import math

from typing import List

from matplotlib import pyplot as plt

from calculations import initialize, lambda_eps_tensor_calculate, effective_stiffness_calculate, \
    effective_compliance_calculate, effective_compliance_calculate_kanaun_levin_method, \
    effective_stiffness_calculate_kanaun_levin_method, effective_compliance_calculate_linear_maxwell_method, \
    effective_stiffness_calculate_linear_maxwell_method, effective_compliance_calculate_mori_tanaka_method, \
    effective_stiffness_calculate_mori_tanaka_method, effective_stiffness_calculate_maxwell_method, \
    effective_compliance_calculate_maxwell_method, lambda_sig_tensor_calculate


def tensor_by_volume_fraction(volume: float, n: int, fi_list: List[float], matrix_const: List,
                                        inhomo_const: List, method, orientation: bool = False):
    '''
    Зависимость коэффициентов тензора модулей упругости от объемной доли
    :param orientation:
    :param volume: Репрезентативный элемент объема
    :param n: Количество неоднородностей
    :param fi_list: Список объемных долей
    :param matrix_const: Коэффициенты Ламе матрицы
    :param inhomo_const: Коэффициенты Ламе неоднородности
    '''

    b_list = [3 / 4 * volume / n * fi / math.pi / 1 for fi in fi_list]
    Tensor_eff_list_plot = []

    for b in b_list:
        structure = initialize(matrix_const, [[1, b] for _ in range(n)], [inhomo_const for _ in range(n)],
                               ['spheroid' for _ in range(n)], orientation)
        if method in [effective_stiffness_calculate_maxwell_method, effective_stiffness_calculate, effective_stiffness_calculate_mori_tanaka_method, effective_stiffness_calculate_linear_maxwell_method, effective_stiffness_calculate_kanaun_levin_method]:
            lambda_tensors = lambda_eps_tensor_calculate(structure)
        else:
            lambda_tensors = lambda_sig_tensor_calculate(structure)
        Tensor_eff_list_plot.append(method(structure, lambda_tensors, volume))

    Tensor_list = []
    FI_list = []

    for tensor in Tensor_eff_list_plot:
        for i, component in enumerate(tensor):
            if i == 0:
                print('компоненты', component)
                Tensor_list.append(component)
            elif i == 3:
                FI_list.append(component)
                print('доля', component)

    smth = zip(*Tensor_list)
    num = 1
    for el in smth:
        if method in [effective_stiffness_calculate_maxwell_method, effective_stiffness_calculate,
                      effective_stiffness_calculate_mori_tanaka_method,
                      effective_stiffness_calculate_linear_maxwell_method,
                      effective_stiffness_calculate_kanaun_levin_method]:
            plt.plot(FI_list, list(el), label=f'C_{num}')
        else:
            plt.plot(FI_list, list(el), label=f'S_{num}')
        num += 1
    if method in [effective_stiffness_calculate, effective_compliance_calculate]:
        plt.title('Без учета взаимодействия')
    elif method in [effective_stiffness_calculate_kanaun_levin_method, effective_compliance_calculate_kanaun_levin_method]:
        plt.title('Учет взаимодействия методом Канауна-Левина')
    elif method in [effective_stiffness_calculate_mori_tanaka_method, effective_compliance_calculate_mori_tanaka_method]:
        plt.title('Учет взаимодействия методом Мори-Танаки')
    elif method in [effective_stiffness_calculate_maxwell_method, effective_compliance_calculate_maxwell_method]:
        plt.title('Учет взаимодействия методом Максвелла')
    else:
        plt.title('Учет взаимодействия линеаризованным методом Максвелла')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    plt.xlabel('Объемная доля')
    plt.show()