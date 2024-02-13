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
n = 5
fi_list = [i / 10 for i in range(1, 10)]
print(fi_list)

volumes = [spheroid_inhomo.volume * n / fi for fi in fi_list]
print(volumes)
lame_coefficients_list = [[100 / 10 ** i, 100 / 10 ** i] for i in range(5)]
# print(lame_coefficients_list)
C_eff_list = []
for i in lame_coefficients_list:
    structure = initialize([1, 1], [[1, 0.001] for _ in range(n)], [i for _ in range(n)], ['spheroid' for _ in range(n)])
    lambda_tensors = lambda_tensor_calculate(structure)
    C_eff_list.append(effective_stiffness_calculate(structure, lambda_tensors, volumes[-1]))



C_eff_list_plot = []
for volume in volumes:
    structure = initialize([1, 1], [[1, 0.001] for _ in range(n)], [[7, 15] for _ in range(n)],
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
        elif i == 2:
            FI_list.append(component)

smth = zip(*C_list)
num = 1
for el in smth:
    plt.plot(FI_list, list(el), label=f'C_{num}')
    num += 1
plt.legend()
plt.grid(color='gray', linestyle='-', linewidth=0.1)
plt.show()