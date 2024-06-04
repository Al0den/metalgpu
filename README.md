# Metal GPU

This is a simple wrapper around Apple's Metal API, to run compute kernels from python, with full control over buffers and methods

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

buffer1 = instance.createBuffer(buffer_size, 0, buffer_type) # Create a shared gpu-cpu buffer. Can be accessed as a numpy array from buffer1.contents
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
When tested using performance.py:

| Function | CPU Compute Time | GPU Compute Time |
|---|---|---|
| Calculating 100 million cos values  | 36.514s  | 0.131s |

## Documentation
WIP

## Credits
- [MyMetalKernel.py](https://gist.github.com/alvinwan/f7bb0cdd26c018f40052f9944fc5c679/revisions) Didn't manage to get this to work, overcomplicated for python code
