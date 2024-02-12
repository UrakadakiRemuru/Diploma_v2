from typing import List, Annotated
from tensor_classes.tensors import ElasticStiffnessTensor, ResultTransverselyIsotropicTensor


def addition(tensors: List[ElasticStiffnessTensor], sign: bool = True) -> ResultTransverselyIsotropicTensor:
    """Производит сложение или вычитание некоторого количества тензоров жесткости.
    :param tensors: Список тензоров, над которыми будет производится операция.
    :param sign: Логическое значение, определяющее знак операции: для сложения - True, для вычетания - False. По умолчанию True.
        :return: Тензор, получившийся в результате сложения или вычитания."""

    prev_tensor = 0
    for i, tensor in enumerate(tensors):
        length = len(tensor.components())
        if length is not True:
            raise Exception('Тензор не может быть пустым.')
        if i == 0:
            prev_tensor = length
        if i > 0 and length != prev_tensor:
            raise Exception('Тензоры имеют разную размерность.')

    result_tensor: ResultTransverselyIsotropicTensor = ResultTransverselyIsotropicTensor([0 for i in range(6)])
    result: List[float] = []
    for i in range(len(tensors[0].components())):
        sum = 0

        for tensor in tensors:
            if sign:
                sum += tensor.components()[i]
            else:
                sum -= tensor.components()

        result.append(sum)

    result_tensor.components = result

    return result_tensor


def double_dot_product(tensors: Annotated[List[ElasticStiffnessTensor | ResultTransverselyIsotropicTensor], 2]) -> ResultTransverselyIsotropicTensor:
    a1, a2, a3, a4, a5, a6 = tensors[0].components()
    b1, b2, b3, b4, b5, b6 = tensors[1].components()
    result = [2 * a1 * b1 + a3 * b4,
              a2 * b2,
              2 * a1 * b3 + a3 * b6,
              2 * a4 * b1 + a6 * b4,
              0.5 * a5 * b5,
              a6 * b6 + 2 * a4 * b3]

    RTIT = ResultTransverselyIsotropicTensor([0 for i in range(6)])
    RTIT.components = result

    return RTIT
