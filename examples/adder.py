import metalgpu

instance = metalgpu.Interface()  # Initialise the metal instance
shader_string = """
#include <metal_stdlib>

using namespace metal;

kernel void adder(device int *arr1 [[buffer(0)]],
        device int *arr2 [[buffer(1)]],
        device int *arr3 [[buffer(2)]],
        uint id [[thread_position_in_grid]]) {
    arr3[id] = arr2[id] + arr1[id];
}
"""
instance.load_shader_from_string(shader_string)
instance.set_function("adder")

buffer_size = 100000  # Number of items in the buffer
buffer_type = "int"

buffer1 = instance.create_buffer(buffer_size, buffer_type)
buffer2 = instance.create_buffer(buffer_size, buffer_type)
buffer3 = instance.create_buffer(buffer_size, buffer_type)

buffer1.contents[:] = [i for i in range(buffer_size)]
buffer2.contents[:] = [i for i in range(buffer_size)]

instance.run_function(buffer_size, [buffer1, buffer2, buffer3])

assert(all(buffer3.contents == [i * 2 for i in range(buffer_size)]))

buffer1.release()
buffer2.release()
buffer3.release()
