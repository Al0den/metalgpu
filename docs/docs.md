# MetalGPU - Documentation

## Interface

An Interface is the wrapper around the metal functions and their associated buffers. At no point in time should two interface co-exist, as the C extension only allows a single one present at a time. 

To delete an Interface, simply use `del interface` and it will automatically free up all buffers

### Interface.load_shader(shaderPath)
Load a shader from a specific file. Can be changed at any time.
- shaderPath: The path to the .metal file that will be loaded

### Interface.load_shader_from_string(shaderString)
Load a shader from a string. Can be changed at any time
- shaderString: A string that can be resolved as a metal shader

### Interface.set_function(functionName)
Set the function that will be ran when Interface.run_function is called.
- functionName: The name, as presented in the metal shader, of the function

### Interface.create_buffer(bufferSize, bufferType)
Returns a new buffer, created to hold bufferSize elements of type bufferType. The initial content of the buffer is unspecified. 
- bufferSize: The number of elements the buffer will be able to hold
- bufferType: The type of items that will be used. Can be a ctype or a string, that will then be resolved to a ctype

### Interface.array_to_buffer(array)
Returns a new buffer, and copies the array content to said buffer
- array: A numpy array or python list that will be copied into the buffer 

### Interface.run_function(numThreads, buffers)
Runs the currently set function
- numThreads: The number of gpu threads that will be started
- buffers: A list of buffers that will be sent to the GPU. The first element of the list will be associated to buffer nubmer 0, second to 1, etc... If you do not want to associate a buffer to the nth slot, use a None in the string, and continue with the following buffers

## Buffer

A buffer is a shared part of memory between the GPU and CPU. It is the only way to transfer data to a metal shader. 

It can be destroyed using `del buffer`

### Buffer.contents()
Returns the contents of the buffer as a numpy array. 

### Buffer.release()
Frees up the buffers memory. Is automatically called on buffer destruction

### Buffer.interface
The interface the buffer was created from