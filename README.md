# Metal GPU

This is a simple wrapper around Apple's Metal API, to run compute kernels from python, with full control over buffers and methods. No copying behind the scenes, and raw access to the buffers as numpy arrays

## Installing
Simply run `pip install metalgpu` to download latest release, and this should be sufficient

If any errors relating to binaries occur, you can recompile them by downloading this repo, and looking into metal-gpu-c and it's associated Makefile. Moving the `lib.so` file to the correct place should fix the issue

## Examples

**main.py**
```python
import metalgpu
import ctypes
import numpy as np

instance = metalgpu.createInterface() # Initialise the metal instance
instance.loadShader("./shader.metal") # Path to the metal shader
instance.setFunction("adder") # Name of the function that will be ran (Can be changed at any time)

buffer_size = 100000 # Number of items in the buffer
buffer_type = ctypes.c_int # Types of the items inside the buffer

buffer1 = instance.createBuffer(buffer_size, 0, buffer_type) # Create a shared gpu-cpu buffer.
buffer2 = instance.createBuffer(buffer_size, 1, buffer_type)
buffer3 = instance.createBuffer(buffer_size, 2, buffer_type)

for i in range(buffer_size):
    buffer1.contents[i] = i
    buffer2.contents[i] = int(np.sqrt(i))

instance.runFunction(buffer_size) # Computes i + int(sqrt(i))

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
| Calculating 100 million cos values  | 36.514s  | 0.131s |

Note: The GPU compute is almost as fast computing 1 million or 10 calculations, being limited by throughput to about 0.001s minimum per function run.
## Documentation

The available commands are, as of right now:
- `createInterface()`, creates the Metal instance
- `instance.loadShader(shaderPath)`, loads the shader file
- `instance.setFunction(functionName)`, sets the function that will be used. This can be changed at any time
- `instance.createBuffer(numItems, bufferNum, bufferType)`, creates a shared buffer. bufferNum refers to the buffer identifier for the shader. bufferType should be a ctype, similar to the examples.
- `buffer.release()`, free up the buffer. You should always free up memory that you will not use again.
- `buffer.contents`, a numpy array vision of the buffer. It can be manipulated as a numpy array, however keep in mind that it should still be readable to the gpu. No copying is going on behind the scenes
- `instance.runFunction(numThreads)`, runs the set function, starting up 'numthreads' different threads.

## Credits
- [MyMetalKernel.py](https://gist.github.com/alvinwan/f7bb0cdd26c018f40052f9944fc5c679/revisions) Didn't manage to get this to work, overcomplicated for python code
- [metalcompute](https://github.com/baldand/py-metal-compute) Although similar, performs lots of array copies instead of buffer management, and has some memory leaks. 
