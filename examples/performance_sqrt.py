import metalgpu
import ctypes
import numpy as np
import time

instance = metalgpu.Interface()

shader_str = """
#include <metal_stdlib>

using namespace metal;

kernel void sqrt_func(device float* arr[[buffer(0)]], uint id [[thread_position_in_grid]]) {
    arr[id] = sqrt(arr[id]);
}
"""

instance.load_shader_from_string(shader_str)
instance.set_function("sqrt_func")


buffer_size = 1000000
buffer_type = ctypes.c_float

buffer1 = instance.create_buffer(buffer_size, buffer_type)

buffer1.contents[:] = [i for i in range(buffer_size)]

np_start = time.time()
out_np = [np.sqrt(i) for i in buffer1.contents]
np_end = time.time()

gpu_start = time.time()
instance.run_function(buffer_size, [buffer1])
gpu_end = time.time()

print("Sqrt test - CPU time: ", np_end - np_start)
print("Sqrt test - GPU time: ", gpu_end - gpu_start)

assert(np.allclose(buffer1.contents, out_np, atol=1e-5))

buffer1.release()





