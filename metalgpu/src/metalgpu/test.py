from interface_class import Interface

import numpy as np

interface = Interface()

arr = np.array([i for i in range(5000)], dtype=np.int32)
arr2 = np.array([2 * i for i in range(5000)], dtype=np.int32)

buff1 = interface.array_to_buffer(arr)
buff2 = interface.array_to_buffer(arr2)

outbuf = buff1 - buff2

print(outbuf.contents)

