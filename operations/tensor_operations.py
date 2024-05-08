from typing import List, Annotated
from tensor_classes.tensors import ElasticStiffnessTensor, ResultTransverselyIsotropicTensor, TransverselyIsotropicTensor

def multiplication(a: float, tensor: ElasticStiffnessTensor | ResultTransverselyIsotropicTensor | TransverselyIsotropicTensor) -> ResultTransverselyIsotropicTensor:
    '''Тензорное умножение вектора на константу.
    :param a: Целочисленное значение(постоянная).
    :param tensor: Тензор.
        :return: Тензор, получившийся в результате умножения.'''

    result: List[float] = [a * i for i in tensor.components]
    return ResultTransverselyIsotropicTensor(result)
def addition(tensors: List[ElasticStiffnessTensor | ResultTransverselyIsotropicTensor | TransverselyIsotropicTensor], sign: bool = True) -> ResultTransverselyIsotropicTensor:
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

    return ResultTransverselyIsotropicTensor(result)


def double_dot_product(tensors: Annotated[List[ElasticStiffnessTensor | ResultTransverselyIsotropicTensor], 2]) -> ResultTransverselyIsotropicTensor:
    '''
    Осуществляет операцию свертки двух изотропных тензоров четвертого ранга, разложенных в тензорном базисе.
    :param tensors: Массив из двух тензоров, подлежащих свертыванию.
        :return: Тензор, полученный в результате свертки.
    '''
    a1, a2, a3, a4, a5, a6 = tensors[0].components
    b1, b2, b3, b4, b5, b6 = tensors[1].components
    result: List[float] = [2 * a1 * b1 + a3 * b4,
              a2 * b2,
              2 * a1 * b3 + a3 * b6,
              2 * a4 * b1 + a6 * b4,
              0.5 * a5 * b5,
              a6 * b6 + 2 * a4 * b3]

    return ResultTransverselyIsotropicTensor(result)

def avg_over_orientations(tensor: TransverselyIsotropicTensor) -> ResultTransverselyIsotropicTensor:
    c = tensor.components
    return ResultTransverselyIsotropicTensor([
        1 / 15 * (7 * c[0] + c[1] + 3 * c[2] + 3 * c[3] + 0.5 * c[4] + 2 * c[5]),
        1 / 15 * (2 * c[0] + 6 * c[1] - 2 * c[2] - 2 * c[3] + 3 * c[4] + 2 * c[5]),
        1 / 15 * (6 * c[0] - 2 * c[1] + 4 * c[2] + 4 * c[3] - c[4] + c[5]),
        1 / 15 * (6 * c[0] - 2 * c[1] + 4 * c[2] + 4 * c[3] - c[4] + c[5]),
        1 / 15 * (4 * c[0] + 12 * c[1] - 4 * c[2] - 4 * c[3] + 6 * c[4] + 4 * c[5]),
        1 / 15 * (8 * c[0] + 4 * c[1] + 2 * c[2] + 2 * c[3] + 2 * c[4] + 3 * c[5]),
    ])
