import metalgpu as mg
import numpy as np

interface = mg.Interface()

arr_size = 10000

arr1 = np.random.rand(arr_size).astype(np.float32) * 100
buf1 = interface.array_to_buffer(arr1)

buf2 = mg.sqrt(buf1)
buf3 = mg.cos(buf1)
buf4 = mg.sin(buf1)

assert(np.allclose(np.sqrt(arr1), buf2.contents, atol=1e-5))
assert(np.allclose(np.cos(arr1), buf3.contents, atol=1e-5))
assert(np.allclose(np.sin(arr1), buf4.contents, atol=1e-5))




