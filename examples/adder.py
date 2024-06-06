import metalgpu
import ctypes
import numpy as np

instance = metalgpu.Interface() # Initialise the metal instance
shader_string = """
#include <metal_stdlib>

using namespace metal;

kernel void adder(device int *arr1 [[buffer(0)]], device int *arr2 [[buffer(1)]], device int *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    arr3[id] = arr2[id] + arr1[id];
}
"""
instance.load_shader_from_string(shader_string)
instance.set_function("adder") # Name of the function that will be ran (Can be changed at any time)

buffer_size = 100000 # Number of items in the buffer
buffer_type = ctypes.c_int # Types of the items inside the buffer

buffer1 = instance.create_buffer(buffer_size, buffer_type) # Create a shared gpu-cpu buffer. Can be accessed as a numpy array from buffer1.contents
buffer2 = instance.create_buffer(buffer_size, buffer_type)
buffer3 = instance.create_buffer(buffer_size, buffer_type)

for i in range(buffer_size):
    buffer1.contents[i] = i
    buffer2.contents[i] = int(np.sqrt(i))

instance.run_function(buffer_size, [buffer1, buffer2, buffer3]) # Computes i + sqrt(i)

for i in range(buffer_size):
    assert(buffer3.contents[i] == i + int(np.sqrt(i))) 

buffer1.release()
buffer2.release()
buffer3.release()
