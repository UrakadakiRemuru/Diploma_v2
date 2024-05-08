from typing import List, Annotated
from tensors import IsotropicTensor


def multiplication(a: float, tensor: IsotropicTensor) -> IsotropicTensor:
    '''Тензорное умножение вектора на константу.
    :param a: Целочисленное значение(постоянная).
    :param tensor: Тензор.
        :return: Тензор, получившийся в результате умножения.'''

    result: Annotated[List[float], 2] = [a * i for i in tensor.components]
    return IsotropicTensor(result)


def addition(tensors: List[IsotropicTensor], sign: bool = True) -> IsotropicTensor:
    """Производит сложение или вычитание некоторого количества тензоров жесткости.
    :param tensors: Список тензоров, над которыми будет производится операция.
    :param sign: Логическое значение, определяющее знак операции: для сложения - True, для вычетания - False. По умолчанию True.
        :return: Тензор, получившийся в результате сложения или вычитания."""

    prev_tensor = 0
    for i, tensor in enumerate(tensors):
        length = len(tensor.components)
        if not length:
            raise Exception('Тензор не может быть пустым.')
        if i == 0:
            prev_tensor = length
        if i > 0 and length != prev_tensor:
            raise Exception('Тензоры имеют разную размерность.')

    result: List[float] = []
    for i in range(len(tensors[0].components)):
        sum = 0

        for j, tensor in enumerate(tensors):
            if sign:
                sum += tensor.components[i]
            else:
                if j == 0:
                    sum = tensor.components[i]
                else:
                    sum -= tensor.components[i]

        result.append(sum)

    return IsotropicTensor(result)

def double_dot_product(tensors: List[IsotropicTensor]) -> IsotropicTensor:

    result = []
    for i in range(len(tensors[0].components)):
        component = 1
        for j, tensor in enumerate(tensors):
            if j != 0 and tensor == tensors[j - 1]:
                pass
            else:
                component *= tensor.components[i]
        result.append(component)

    return IsotropicTensor(result)
