import numpy as np
from interface_class import Interface
import operators as ops

interface = Interface()

arr1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
buf1 = interface.array_to_buffer(arr1)

buf2 = ops.sqrt(buf1)
print(buf2.contents)
