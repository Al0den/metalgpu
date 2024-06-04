import metalgpu
import ctypes
import numpy as np
import time

instance = metalgpu.Interface()
instance.load_shader("./shader.metal")
instance.set_function("sqrt_func")

buffer_size = 10000000
buffer_type = ctypes.c_float

buffer1 = instance.create_buffer(buffer_size, buffer_type)

for i in range(buffer_size):
    buffer1.contents[i] = float(i)

np_start = time.time()
out_np = [np.sqrt(i) for i in buffer1.contents]
np_end = time.time()

gpu_start = time.time()
instance.run_function(buffer_size, [buffer1])
gpu_end = time.time()

print("CPU time: ", np_end - np_start)
print("GPU time: ", gpu_end - gpu_start)

assert(np.allclose(buffer1.contents, out_np, atol=1e-5))

buffer1.release()





