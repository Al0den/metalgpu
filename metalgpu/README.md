## Metal GPU

This is a simple wrapper around Apple's Metal API, to run compute kernels from python, with full control over buffers and methods. No copying behind the scenes, and raw access to the buffers as numpy arrays

## Installing
Simply run `pip install metalgpu` to download latest release, and this should be sufficient

If any errors relating to binaries occur, you can recompile them by downloading this repo, and looking into metal-gpu-c and it's associated Makefile. Moving the `lib.so` file to the correct place should fix the issue

## Examples

**main.py**
```python
from metalgpu import Interface
import ctypes
import numpy as np

instance = Interface() # Initialise the metal instance
instance.load_shader("./shader.metal") # Path to the metal shader
instance.set_function("adder") # Name of the function that will be ran (Can be changed at any time)

buffer_size = 100000 # Number of items in the buffer
buffer_type = ctypes.c_int # Types of the items inside the buffer

buffer1 = instance.create_buffer(buffer_size, buffer_type) # Create a shared gpu-cpu buffer.
buffer2 = instance.create_buffer(buffer_size, buffer_type)
buffer3 = instance.create_buffer(buffer_size, buffer_type)

for i in range(buffer_size):
    buffer1.contents[i] = i
    buffer2.contents[i] = int(np.sqrt(i))

instance.run_function(buffer_size, [buffer1, buffer2, buffer3]) # Computes i + int(sqrt(i))

for i in range(buffer_size):
    assert(buffer3.contents[i] == i + int(np.sqrt(i))) 

buffer1.release()
buffer2.release()
buffer3.release()
```
**shader.metal**
```
#include <metal_stdlib>

using namespace metal;

kernel void adder(device int* arr1 [[buffer(0)]], device int* arr2 [[buffer(1)]], device int* arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    arr3[id] = arr2[id] + arr1[id];
};
```

## Performance
When tested using performance.py, on Apple Silicon M1 Pro, base specs:

| Function | CPU Compute Time | GPU Compute Time |
|---|---|---|
| Calculating 10 million cos values  | 3.553s  | 0.0100s |
| Calculating 10 million square roots  | 3.737s | 0.00694s |

Note: The GPU compute is almost as fast computing 1 million or 10 calculations, being limited by throughput to about 0.001s minimum per function run.
## Documentation

The available commands are, as of right now:
- `metalgpu.Interface()`, creates the Metal instance
- `instance.load_shader(shaderPath)`, loads the shader file
- `instance.load_shader_from_str(string)`, loads a shader from a string
- `instance.set_function(functionName)`, sets the function that will be used. This can be changed at any time
- `instance.create_buffer(numItems, bufferType)`, creates a shared buffer. bufferType should be a ctype, similar to the examples.
- `buffer.release()`, free up the buffer. You should always free up memory that you will not use again.
- `buffer.contents`, a numpy array vision of the buffer. It can be manipulated as a numpy array, however keep in mind that it should still be readable to the gpu. No copying is going on behind the scenes
- `instance.run_function(numThreads, buffers)`, runs the set function, starting up 'numthreads' different threads. buffers should be a list of buffers, with the first being referenced as buffer 0 in metal. If you want to "skip" a buffer number, as to use buffer 0 and 2, do [buff0, None, buff2]
- `instance.array_to_buffer(array)`, creates a buffer and copies the numpy array to the buffer

## Credits
- [MyMetalKernel.py](https://gist.github.com/alvinwan/f7bb0cdd26c018f40052f9944fc5c679/revisions) Didn't manage to get this to work, overcomplicated for python code
- [metalcompute](https://github.com/baldand/py-metal-compute) Although similar, performs lots of array copies instead of buffer management, and has some memory leaks. 
 A Python wrapper for Apple's Metal API
