from calculations import effective_compliance_calculate_kanaun_levin_method, \
    effective_stiffness_calculate_kanaun_levin_method, effective_compliance_calculate_linear_maxwell_method, \
    effective_stiffness_calculate_linear_maxwell_method, effective_compliance_calculate_mori_tanaka_method, \
    effective_stiffness_calculate_mori_tanaka_method, effective_stiffness_calculate_maxwell_method, \
    effective_compliance_calculate_maxwell_method
from operations.operations import E_nu_to_lam_mu
from plots_by_volume_fraction import tensor_by_volume_fraction
from yield_stress import yield_stress_by_volume_fraction

n = 5  # количество неоднородностей

fi_list = [i / 100 for i in range(0, 101)]  # объемная доля

volume = 1  # значение объема репрезентативного объема

lame_coefficients_list = [[100 / 10 ** i, 100 / 10 ** i] for i in range(5)]

# 1
matrix_const = E_nu_to_lam_mu(1000, 0.3)
inhomo_const = E_nu_to_lam_mu(10, 0.3)

# # 2
# matrix_const = E_nu_to_lam_mu(10, 0.3)
# inhomo_const = E_nu_to_lam_mu(1000, 0.3)

# # 3
# matrix_const = inhomo_const = E_nu_to_lam_mu(1000, 0.3)



# print('Без учета взаимодействия.')
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate)
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate)
# print('------------------------------------------------')
#
# print('Учет взаимодействия методом Мори-Танаки.')
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_mori_tanaka_method)
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_mori_tanaka_method)
# print('------------------------------------------------')
#
#
# print('Учет взаимодействия методом Максвелла.')
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_maxwell_method)
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_maxwell_method)
# print('------------------------------------------------')
#
#
# print('Учет взаимодействия линеаризованным методом Максвелла.')
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_linear_maxwell_method)
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_linear_maxwell_method)
# print('------------------------------------------------')
#
#
# print('Учет взаимодействия методом Канауна-Левина.')
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_kanaun_levin_method)
# tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_kanaun_levin_method)
# print('------------------------------------------------')

print('Осредение по ориентациям.')

print('Метод Мори-Танаки')
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_mori_tanaka_method, True)
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_mori_tanaka_method, True)
print('------------------------------------------------')


print(' Метод Максвелла')
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_maxwell_method, True)
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_maxwell_method, True)
print('------------------------------------------------')


print('Линеаризованный метод Максвелла')
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_linear_maxwell_method, True)
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_linear_maxwell_method, True)
print('------------------------------------------------')


print('Метод Канауна-Левина')
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_stiffness_calculate_kanaun_levin_method, True)
tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const, effective_compliance_calculate_kanaun_levin_method, True)
print('------------------------------------------------')

print('Эффективный предел текучести.')
yield_stress_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
print('------------------------------------------------')

