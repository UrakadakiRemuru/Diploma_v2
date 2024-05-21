import math

from calculations import effective_compliance_calculate_kanaun_levin_method, \
    effective_stiffness_calculate_kanaun_levin_method, effective_compliance_calculate_linear_maxwell_method, \
    effective_stiffness_calculate_linear_maxwell_method, effective_compliance_calculate_mori_tanaka_method, \
    effective_stiffness_calculate_mori_tanaka_method, effective_stiffness_calculate_maxwell_method, \
    effective_compliance_calculate_maxwell_method, effective_stiffness_calculate, effective_compliance_calculate
from operations.operations import E_nu_to_lam_mu, K_mu_to_lam_mu
from plots_by_volume_fraction import tensor_by_volume_fraction
from yield_stress import yield_stress_by_volume_fraction

# n = 15  # количество неоднородностей
# a = 1
# b = 0.45
#
# volume = 4 / 3 * n * math.pi * a ** 2 * b  # значение объема репрезентативного объема
# size_list = [a, b, n]  # объемная доля
#
#
# lame_coefficients_list = [[100 / 10 ** i, 100 / 10 ** i] for i in range(5)]
#
# # 1
# matrix_const = E_nu_to_lam_mu(1000, 0.3)
# inhomo_const = E_nu_to_lam_mu(10, 0.3)

# # 2
# matrix_const = E_nu_to_lam_mu(10, 0.3)
# inhomo_const = E_nu_to_lam_mu(1000, 0.3)

# # 3
# matrix_const = inhomo_const = E_nu_to_lam_mu(1000, 0.3)



# print('Без учета взаимодействия.')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate)
# print('------------------------------------------------')
#
# print('Учет взаимодействия методом Мори-Танаки.')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_mori_tanaka_method)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_mori_tanaka_method)
# print('------------------------------------------------')
#
#
# print('Учет взаимодействия методом Максвелла.')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_maxwell_method)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_maxwell_method)
# print('------------------------------------------------')
#
#
# print('Учет взаимодействия линеаризованным методом Максвелла.')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_linear_maxwell_method)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_linear_maxwell_method)
# print('------------------------------------------------')
#
#
# print('Учет взаимодействия методом Канауна-Левина.')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_kanaun_levin_method)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_kanaun_levin_method)
# print('------------------------------------------------')

# print('Осредение по ориентациям.')
#
# print('Метод Мори-Танаки')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_mori_tanaka_method, True)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_mori_tanaka_method, True)
# print('------------------------------------------------')
#
#
# print(' Метод Максвелла')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_maxwell_method, True)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_maxwell_method, True)
# print('------------------------------------------------')
#
#
# print('Линеаризованный метод Максвелла')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_linear_maxwell_method, True)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_linear_maxwell_method, True)
# print('------------------------------------------------')
#
#
# print('Метод Канауна-Левина')
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_stiffness_calculate_kanaun_levin_method, True)
# tensor_by_volume_fraction(volume, size_list, matrix_const, inhomo_const, effective_compliance_calculate_kanaun_levin_method, True)
# print('------------------------------------------------')
n = 20  # количество неоднородностей
volume = 0.0001
g_list = [0.15, 0.2, 0.3, 0.4, 0.6, 0.7, 1]
type: str = 'spheroid'
matrix_const = K_mu_to_lam_mu(77.9 * 10 ** 9, 24.9 * 10 ** 9)
print('do', matrix_const)
inhomo_const = K_mu_to_lam_mu(0.0001, 0.0001)
print('Эффективный предел текучести.')
yield_stress_by_volume_fraction(volume, n, g_list, matrix_const, inhomo_const, True)
print('------------------------------------------------')

