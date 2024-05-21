import math
from typing import List

from matplotlib import pyplot as plt

from calculations import initialize, lambda_eps_tensor_calculate, effective_stiffness_calculate_maxwell_method, \
    lambda_sig_tensor_calculate, effective_compliance_calculate_maxwell_method


def m_stiffness_tensor_by_lame(lame_coefficients_list: List[List], n: int, volume: float, matrix_const: List[float], inhomo_size: List[float]):
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
        res = effective_stiffness_calculate_maxwell_method(structure, lambda_tensors, volume)
        C_eff_list.append(res)

    # for i in C_eff_list:
    #     print(i)

    return C_eff_list

def m_stiffness_tensor_by_volume_fraction(volume: float, n: int, g_list: list[float], matrix_const: List,
                                        inhomo_const: List, type: str):
    '''
    Зависимость коэффициентов тензора модулей упругости от объемной доли
    :param volume: Репрезентативный элемент объема
    :param n: Количество неоднородностей
    :param fi_list: Список объемных долей
    :param matrix_const: Коэффициенты Ламе матрицы
    :param inhomo_const: Коэффициенты Ламе неоднородности
    '''
    E_list_sphere = []
    mu_list_sphere = []
    FI_list_sphere = []
    r_max = (3 / 4 * volume / n / math.pi) ** (1 / 3)
    for i in range(0, 101):
        r = r_max / 100 * i
        structure = initialize(matrix_const, [[r] for _ in range(n)],
                               [inhomo_const for _ in range(n)],
                               ['sphere' for _ in range(n)])
        lambda_tensors = lambda_eps_tensor_calculate(structure)
        C_eff, [E, mu], fi = effective_stiffness_calculate_maxwell_method(structure, lambda_tensors, volume)
        E_list_sphere.append(E)
        mu_list_sphere.append(mu)
        FI_list_sphere.append(fi)

    fi_g = []
    E_g = []
    Mu_g = []
    for g in g_list:
        a_max = (3 / 4 * volume / g / n / math.pi) ** (1 / 3)
        C_list = []
        FI_list = []
        E_list = []
        mu_list = []

        for i in range(0, 101):
            a = a_max / 100 * i
            b = g * a
            structure = initialize(matrix_const, [[a, b] for _ in range(n)],
                                   [inhomo_const for _ in range(n)],
                                   [type for _ in range(n)])
            lambda_tensors = lambda_eps_tensor_calculate(structure)
            C_eff, [E, mu], fi = effective_stiffness_calculate_maxwell_method(structure, lambda_tensors, volume)
            C_list.append(C_eff)
            E_list.append(E)
            mu_list.append(mu)
            FI_list.append(fi)
        fi_g.append(FI_list)
        E_g.append(E_list)
        Mu_g.append(mu_list)

    strain_E = [0.0008222435282837968, 0.0007915627996164909, 0.0007900287631831257, 0.0007915627996164909,
         0.0004494726749760307]
    stress_E = [30.774748923959827, 26.25538020086083, 25.03586800573888, 20.73170731707317, 10.903873744619798]
    l, m = matrix_const
    print('константы', l, m)
    E_list_e = [s * 10 ** 6 / e / m / (3 * l + 2 * m) * (l + m) for e, s in zip(strain_E, stress_E)]
    fi_list_e = [0.048, 0.113, 0.15, 0.221, 0.268]
    mu_list_e = [(1 - fi) ** 2 for fi in fi_list_e]

    fig = plt.figure(figsize=(12, 6))

    # Создаем первый сабплот
    ax1 = fig.add_subplot(1, 2, 1)
    for i, g in enumerate(g_list):
        ax1.plot(fi_g[i], E_g[i], label=r'$\gamma$ = ' + str(g))
    ax1.plot(FI_list_sphere, E_list_sphere, label=r'$\gamma$ = 1')
    ax1.plot(fi_list_e, E_list_e, label='Эксперимент', marker='*', linestyle='')
    ax1.set_xlabel(r'$\phi$')
    ax1.set_ylabel(r'$\frac{E_eff}{E_0}$') #\frac{\lambda_eff}{\lambda_0}
    ax1.legend()
    ax1.grid(True)

    # Создаем второй сабплот
    ax2 = fig.add_subplot(1, 2, 2)
    for i, g in enumerate(g_list):
        ax2.plot(fi_g[i], Mu_g[i], label=r'$\gamma$ = ' + str(g))
    ax2.plot(FI_list_sphere, mu_list_sphere, label=r'$\gamma$ = 1')
    ax2.plot(fi_list_e, mu_list_e, label='Эксперимент', marker='*', linestyle='')
    ax2.set_xlabel(r'$\phi$')
    ax2.set_ylabel(r'$\frac{\mu_eff}{\mu_0}$')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle('Метод Максвелла')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def m_compliance_tensor_by_volume_fraction(volume: float, n: int, fi_list: List[float], matrix_const: List,
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
        S_eff_list_plot.append(effective_compliance_calculate_maxwell_method(structure, lambda_tensors, volume))

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
    plt.title('Учет взаимодействия методом Максвелла')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.1)
    plt.xlabel('Объемная доля')
    plt.show()