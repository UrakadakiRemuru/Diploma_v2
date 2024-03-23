from operations.operations import E_nu_to_lam_mu

from maxwell_method import m_compliance_tensor_by_volume_fraction
from linear_maxwell_method import lm_compliance_tensor_by_volume_fraction

n = 5  # количество неоднородностей

fi_list = [i / 10 for i in range(0, 11)]  # объемная доля

volume = 1  # значение объема репрезентативного объема

lame_coefficients_list = [[100 / 10 ** i, 100 / 10 ** i] for i in range(5)]

# 1
matrix_const = E_nu_to_lam_mu(1000, 0.3)
inhomo_const = E_nu_to_lam_mu(10, 0.3)
#
# # 2
# matrix_const = E_nu_to_lam_mu(10, 0.3)
# inhomo_const = E_nu_to_lam_mu(1000, 0.3)
#
# # 3
# matrix_const = inhomo_const = E_nu_to_lam_mu(1000, 0.3)



print('Без учета взаимодействия.')

print('------------------------------------------------')


print('Учет взаимодействия методом Максвелла.')
m_compliance_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
print('------------------------------------------------')


print('Учет взаимодействия линеаризованным методом Максвелла.')
lm_compliance_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
print('------------------------------------------------')


print('Учет взаимодействия методом Канауна-Левина.')

print('------------------------------------------------')


