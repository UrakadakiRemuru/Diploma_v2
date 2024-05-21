import math
from typing import List

from matplotlib import pyplot as plt

from calculations import initialize, calculate_yield_stress, calculate_stress


def yield_stress_by_volume_fraction(volume: float, n: int, g_list: List[float], matrix_const: List,
                                        inhomo_const: List, orientation: bool = False):
    '''
    Зависимость предела текучести от объемной доли
    :param volume: Репрезентативный элемент объема
    :param size_list: Список объемных долей
    :param matrix_const: Коэффициенты Ламе матрицы
    :param inhomo_const: Коэффициенты Ламе неоднородности
    '''
    yield_stress_list_g = []
    FI_list_g = []
    fi_list = [1 - 0.048, 1 - 0.113, 1 - 0.15, 1 - 0.221, 1 - 0.268]
    for g in g_list:
        yield_stress_list = []
        FI_list = []
        for fi in fi_list:
            a = (3 / 4 * volume * fi / g / n / math.pi) ** (1 / 3)
            b = a * g
            if a == b:
                structure = initialize(matrix_const, [[a] for _ in range(n)], [inhomo_const for _ in range(n)],
                                       ['sphere' for _ in range(n)], orientation)
            else:
                structure = initialize(matrix_const, [[a, b] for _ in range(n)], [inhomo_const for _ in range(n)],
                                       ['spheroid' for _ in range(n)], orientation)
            stress = calculate_stress()
            t, fi, A1, A2 = calculate_yield_stress(structure, volume, stress, matrix_const)
            yield_stress_list.append(t)
            FI_list.append(fi)
        yield_stress_list_g.append(yield_stress_list)
        FI_list_g.append(FI_list)
    markers = ['.', 'o', 'v', 's', 'p', '*', '+', 'x']
    stress = [i * 10 ** 6 for i in [35.43758967001435, 30.631276901004302, 28.909612625538017, 23.45767575322812, 21.233859397417504]]
    fi_list_e = [0.048, 0.113, 0.15, 0.221, 0.268]
    for i, g in enumerate(g_list):
        plt.plot(fi_list_e, yield_stress_list_g[i], marker=markers[i], linestyle='', label=r'\gamma = ' + str(g))
    plt.plot(fi_list_e, stress, marker='x', linestyle='', label='Эксперимент')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\tau_*$')
    plt.title('Эффективный предел текучести')
    plt.legend()
    plt.grid(True)
    plt.show()
