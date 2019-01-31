# -*- coding: utf-8 -*-

from context import oslo_method_python as om
import numpy as np


matrix = np.array([[2, 3, 4], [1, -1, 4], [-1, 1, -5]])
# matrix = np.random.uniform(low=-0.1, high=1, size=(10, 10))

print(matrix)
print(om.fill_negative(matrix, window_size=2))