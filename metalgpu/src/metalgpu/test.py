from interface_class import Interface

instance = Interface()  # Initialise the metal instance
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
# Note: For clearer code, use instance.load_shader(shaderPath) to load a metal file

instance.load_shader_from_string(shader_string)
instance.set_function("adder")

buffer_size = 100000  # Number of items in the buffer
buffer_type = "int"

initial_array = [i for i in range(buffer_size)]

buffer1 = instance.array_to_buffer(initial_array)
buffer2 = instance.array_to_buffer(initial_array)
buffer3 = instance.create_buffer(buffer_size, buffer_type)

instance.run_function(buffer_size, [buffer1, buffer2, buffer3])

assert(all(buffer3.contents == [i * 2 for i in range(buffer_size)]))

buffer1.release()
buffer2.release()
buffer3.release()


