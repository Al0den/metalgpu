import numpy as np
from interface_class import Interface

interface = Interface()

arr1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
buf1 = interface.array_to_buffer(arr1)


buf1.as_int()

print(buf1.contents)
