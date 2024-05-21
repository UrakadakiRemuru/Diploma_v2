import math

from kanaun_levin_method import kl_stiffness_tensor_by_volume_fraction
from mori_tanaka_method import mt_stiffness_tensor_by_volume_fraction, mt_compliance_tensor_by_volume_fraction
from operations.operations import E_nu_to_lam_mu, K_mu_to_lam_mu

from maxwell_method import m_compliance_tensor_by_volume_fraction, m_stiffness_tensor_by_volume_fraction
from linear_maxwell_method import lm_compliance_tensor_by_volume_fraction, lm_stiffness_tensor_by_volume_fraction
from without_interaction import wi_stiffness_tensor_by_volume_fraction, wi_compliance_tensor_by_volume_fraction

n = 20  # количество неоднородностей
volume = 0.0001
g_list = [1]
type: str = 'spheroid'

strain = [0.0010232023010546501, 0.000990987535953979, 0.000975647171620326, 0.0009418983700862896, 0.0009372962607861936]
stress = [35.43758967001435, 30.631276901004302, 28.909612625538017, 23.45767575322812, 21.233859397417504] # 4.8, 11.3, 15.0, 22.1, 26.8 (MPa) yield stress
strain_E = [0.00048475551294343243, 0.0004832214765100671, 0.0004816874400767018, 0.00048015340364333656, 0.00047861936720997126]
stress_E = [18.29268292682927, 16.499282639885223, 15.566714490674318, 13.055954088952653, 11.836441893830703] # 4.8, 11.3, 15.0, 22.1, 26.8 (MPa) yield stress

lame_coefficients_list = [[100 / 10 ** i, 100 / 10 ** i] for i in range(5)]

# 1
matrix_const = K_mu_to_lam_mu(77.9 * 10 ** 9, 24.9 * 10 ** 9)
inhomo_const = K_mu_to_lam_mu(0.0001, 0.0001)

# # 2
# matrix_const = E_nu_to_lam_mu(10, 0.3)
# inhomo_const = E_nu_to_lam_mu(1000, 0.3)

# # 3
# matrix_const = inhomo_const = E_nu_to_lam_mu(1000, 0.3)



# print('Без учета взаимодействия.')
# wi_stiffness_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
# wi_compliance_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
# print('------------------------------------------------')
#
# print('Учет взаимодействия методом Мори-Танаки.')
# mt_stiffness_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
# mt_compliance_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
# print('------------------------------------------------')


print('Учет взаимодействия методом Максвелла.')
m_stiffness_tensor_by_volume_fraction(volume, n, g_list, matrix_const, inhomo_const, type)
# m_compliance_tensor_by_volume_fraction(volume, n, g, matrix_const, inhomo_const)
print('------------------------------------------------')


# print('Учет взаимодействия линеаризованным методом Максвелла.')
# lm_stiffness_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
# lm_compliance_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
# print('------------------------------------------------')


# # print('Учет взаимодействия методом Канауна-Левина.')
# kl_stiffness_tensor_by_volume_fraction(volume, n, fi_list, matrix_const, inhomo_const)
# # print('------------------------------------------------')


