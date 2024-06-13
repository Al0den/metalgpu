[![Python application](https://github.com/Al0den/metalgpu/actions/workflows/python-app.yml/badge.svg)](https://github.com/Al0den/metalgpu/actions/workflows/python-app.yml)

# Metal GPU

This is a simple python library, wrapping Apple's Metal API to run compute kernels from python, with full control over buffers and methods. No copying behind the scenes, and raw access to the buffers as numpy arrays

## Installing
Running `pip install metalgpu` to download latest release. After the first install, you will need to compile the C library.

To do so, simply run in your terminal `python -m metalgpu build`, and let it build the library. This leaves no files behind, apart from the compiled library.

Note: You need to have the git command line to use this tool, otherwise manually compile the folder `metal-gpu-c` and move the output library to the lib folder


## Examples

**main.py**
```python
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
```

## Performance
When tested using performance.py, on Apple Silicon M1 Pro, base specs:

| Function | CPU Compute Time | GPU Compute Time |
|---|---|---|
| Calculating 10 million cos values  | 3.553s  | 0.0100s |
| Calculating 10 million square roots  | 3.737s | 0.00694s |

Note: The GPU compute is almost as fast computing 1 million or 10 calculations, being limited by throughput to about 0.001s minimum per function run.
## Documentation

To view the documentation, simply go to the docs folder and view the `docs.md` file

## Known issues
- None :)

## Credits
- [metalcpp](https://github.com/bkaradzic/metal-cpp) The wrapper from Objective-C to Metal, that is used to interact with Metal
- [MyMetalKernel.py](https://gist.github.com/alvinwan/f7bb0cdd26c018f40052f9944fc5c679/revisions) Didn't manage to get this to work, overcomplicated for python code
- [metalcompute](https://github.com/baldand/py-metal-compute) Although similar, performs lots of array copies instead of buffer management, and has some memory leaks. 
