import metalgpu
import ctypes
import numpy as np
import time

instance = metalgpu.createInterface()
instance.loadShader("./shader.metal")
instance.setFunction("cos_func")

buffer_size = 100000000
buffer_type = ctypes.c_float

buffer1 = instance.createBuffer(buffer_size, 0, buffer_type)
buffer2 = instance.createBuffer(buffer_size, 1, buffer_type)

for i in range(buffer_size):
    buffer1.contents[i] = i/buffer_size

np_start = time.time()
out_np = [np.cos(i) for i in buffer1.contents]
np_end = time.time()

gpu_start = time.time()
instance.runFunction(buffer_size)
gpu_end = time.time()

print("CPU time: ", np_end - np_start)
print("GPU time: ", gpu_end - gpu_start)

assert(np.allclose(buffer2.contents, out_np, atol=1e-5))




