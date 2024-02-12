from typing import List

from tensor_classes.tensors import ElasticStiffnessTensor


class inhomogeneity:

    def __init__(self, size: List[float], const: List[float], type: str):
        self.size = size
        self.EST = ElasticStiffnessTensor(const)
        self.type = ''

    def volume(self):
        pass