# Made to test every single feature, pretty useless

import metalgpu as mg
import numpy as np

print("---Starting all test---")
test_size = 100000

print("Testing buffer functions", end="\r")
interface = mg.Interface()
buf1 = interface.array_to_buffer([i for i in range(test_size)])
buf2 = interface.array_to_buffer([i for i in range(test_size)])

assert(np.all(buf1.contents == [i for i in range(test_size)]))
assert(np.all(buf2.contents == [i for i in range(test_size)]))

buf3 = buf1 + buf2
buf4 = buf1 - buf2

# Avoiding to big integers to get a bigger test_size
del buf2
buf2 = interface.array_to_buffer([3 for _ in range(test_size)])
buf5 = buf1 * buf2

assert(np.all(buf3.contents == [i * 2 for i in range(test_size)]))
assert(np.all(buf4.contents == [0 for _ in range(test_size)]))
assert(np.all(buf5.contents == [i * 3 for i in range(test_size)]))

del buf2
del buf3
del buf4
del buf5

buf2 = interface.array_to_buffer([i for i in range(test_size)])

print("Buffer functions test passed")
print("Testing shader compilation", end="\r")

test_shader = f"""
#include <metal_stdlib>

using namespace metal;

kernel void adder_func(const device int *arr1 [[buffer(0)]], const device int *arr2 [[buffer(1)]], device int *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
    arr3[id] = arr2[id] + arr1[id];
}};
"""
interface.load_shader_from_string(test_shader)
interface.set_function("adder_func")

buf3 = interface.create_buffer(test_size, "int")
interface.run_function(test_size, [buf1, buf2, buf3])

assert(np.all(buf3.contents == [i * 2 for i in range(test_size)]))
del buf3

buf3 = buf1 + buf2

assert(np.all(buf3.contents == [i * 2 for i in range(test_size)]))
del buf3

interface.load_shader_from_string(test_shader)
interface.set_function("adder_func")

buf3 = interface.create_buffer(test_size, "int")
interface.run_function(test_size, [buf1, buf2, buf3])

assert(np.all(buf3.contents == [i * 2 for i in range(test_size)]))
del buf3

print("Shader compilation test passed")
print("Testing operators", end="\r")
buf1_float = buf1.astype("float")
assert(buf1_float.contents.dtype == np.float32)
del buf1_float

buf1_short = buf1.astype("short")
assert(buf1_short.contents.dtype == np.int16)
del buf1_short

del buf1
del buf2

buf1 = interface.array_to_buffer([float(i) for i in range(test_size)])

buf2 = mg.sqrt(buf1)
buf3 = mg.cos(buf1)
buf4 = mg.sin(buf1)

expected2 = [np.sqrt(i) for i in range(test_size)]
expected3 = [np.cos(i) for i in range(test_size)]
expected4 = [np.sin(i) for i in range(test_size)]

assert(np.allclose(buf2.contents, expected2, atol=1e-2))
assert(np.allclose(buf3.contents, expected3, atol=1e-2))
assert(np.allclose(buf4.contents, expected4, atol=1e-2))

print("Operators test passed")

print("---All functions working---")





