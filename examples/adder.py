import metalgpu
import ctypes
import numpy as np

instance = metalgpu.Interface() # Initialise the metal instance
instance.load_shader("./shader.metal") # Path to the metal shader
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

