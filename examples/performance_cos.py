import metalgpu
import numpy as np
import time

instance = metalgpu.Interface()

shader_str = """
#include <metal_stdlib>

using namespace metal;

kernel void cos_func(device float* arr[[buffer(0)]], uint id [[thread_position_in_grid]]) {
    arr[id] = cos(arr[id]);
}
"""

instance.load_shader_from_string(shader_str)
instance.set_function("cos_func")

buffer_size = 1000000

buffer1 = instance.create_buffer(buffer_size, "float")
buffer1.contents[:] = [i/buffer_size for i in range(buffer_size)]

np_start = time.time()
out_np = [np.cos(i) for i in buffer1.contents]
np_end = time.time()

gpu_start = time.time()

max_width = instance.threadExecutionWidth()
thread_size = metalgpu.MetalSize(buffer_size, 1, 1)

instance.run_function(thread_size, [buffer1])
gpu_end = time.time()

print("Cos test - CPU time: ", np_end - np_start)
print("Cos test - GPU time: ", gpu_end - gpu_start)

assert(np.allclose(buffer1.contents, out_np, atol=1e-5))

buffer1.release()





