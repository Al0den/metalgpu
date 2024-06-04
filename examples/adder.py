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

instance.runFunction(buffer_size) # Computes i + sqrt(i)

for i in range(buffer_size):
    assert(buffer3.contents[i] == i + int(np.sqrt(i))) 
