import math
from typing import List

from matplotlib import pyplot as plt

from calculations import initialize, calculate_yield_stress, calculate_stress


def yield_stress_by_volume_fraction(volume: float, n: int, fi_list: List[float], matrix_const: List,
                                        inhomo_const: List):
    '''
    Зависимость предела текучести от объемной доли
    :param volume: Репрезентативный элемент объема
    :param n: Количество неоднородностей
    :param fi_list: Список объемных долей
    :param matrix_const: Коэффициенты Ламе матрицы
    :param inhomo_const: Коэффициенты Ламе неоднородности
    '''

    b_list = [3 / 4 * volume / n * fi / math.pi / 1 for fi in fi_list]
    yield_stress_eff_list_plot = []

    for b in b_list:
        structure = initialize(matrix_const, [[1, b] for _ in range(n)], [inhomo_const for _ in range(n)],
                               ['spheroid' for _ in range(n)])
        stress = calculate_stress()
        yield_stress_eff_list_plot.append(calculate_yield_stress(structure, volume, stress))

    yield_stress_list = []
    FI_list = []

    for pair in yield_stress_eff_list_plot:
        print('предел текучести', pair[0])
        print('доля', pair[1])
        yield_stress_list.append(pair[0])
        FI_list.append(pair[1])

    num = 1
    for yield_stress, fi in zip(yield_stress_list, FI_list):
        plt.plot(FI_list, yield_stress, label=f'\u03C4*')
        num += 1

    plt.title('Эффективный предел текучести')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    plt.xlabel('Объемная доля')
    plt.show()