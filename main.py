import math

from calculations import initialize, lambda_tensor_calculate, effective_stiffness_calculate

from Inhomogeneities.Inhomogeneities import inhomogeneity

import matplotlib.pyplot as plt


lame_coefficients = [1, 2]
# C_0 = ElasticStiffnessTensor(lame_coefficients)
# C_1 = ElasticStiffnessTensor(lame_coefficients)
# I = TransverselyIsotropicTensor([1, 1])
spheroid_inhomo = inhomogeneity([1, 0.001], lame_coefficients, 'spheroid')
print(5 * spheroid_inhomo.volume / 0.025)


# меняем коэффы ламе и смотрим, что происходит

n = 5  # количество неоднородностей

fi_list = [i / 10 for i in range(0, 11)]  # объемная доля

volume = 1  # значение объема репрезентативного объема

lame_coefficients_list = [[100 / 10 ** i, 100 / 10 ** i] for i in range(5)]

C_eff_list = []
for i in lame_coefficients_list:
    structure = initialize([1, 1], [[1, 0.001] for _ in range(n)], [i for _ in range(n)], ['spheroid' for _ in range(n)])
    lambda_tensors = lambda_tensor_calculate(structure)
    res = effective_stiffness_calculate(structure, lambda_tensors, volume)
    C_eff_list.append(res)
    print(res)


# Зависимость коэффициентов тензора модулей упругости от объемной доли

b_list = [3 / 4 * volume / n * fi / math.pi / 1 for fi in fi_list]

C_eff_list_plot = []

for b in b_list:
    structure = initialize([1, 1], [[1, b] for _ in range(n)], [[7, 15] for _ in range(n)],
                           ['spheroid' for _ in range(n)])
    lambda_tensors = lambda_tensor_calculate(structure)
    C_eff_list_plot.append(effective_stiffness_calculate(structure, lambda_tensors, volume))

print(C_eff_list_plot)


C_list = []
FI_list = []
for tensor in C_eff_list_plot:
    for i, component in enumerate(tensor):
        if i == 0:
            C_list.append(component)
        elif i == 3:
            FI_list.append(component)

smth = zip(*C_list)
num = 1
for el in smth:
    plt.plot(FI_list, list(el), label=f'C_{num}')
    num += 1
plt.legend()
plt.grid(color='gray', linestyle='-', linewidth=0.1)
plt.xlabel('Объемная доля')
plt.show()