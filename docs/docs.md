# MetalGPU - Documentation

This is the documentation for the python library metalgpu.

## Interface

An Interface is the wrapper around the metal functions and their associated buffers. At no point in time should two interfaces co-exist, as the C extension only allows a single one present at a time.

To delete an Interface, simply use `del interface` and it will automatically free up all buffers.

### Interface.load_shader(shaderPath)

Load a shader from a specific file. Can be changed at any time.
- shaderPath: The path to the .metal file that will be loaded

### Interface.load_shader_from_string(shaderString)

Load a shader from a string. Can be changed at any time.
- shaderString: A string that can be resolved as a metal shader

### Interface.set_function(functionName)

Set the function that will be run when Interface.run_function is called.
- functionName: The name, as presented in the metal shader, of the function

### Interface.create_buffer(bufferSize, bufferType)

Returns a new buffer, created to hold bufferSize elements of type bufferType. The initial content of the buffer is unspecified.
- bufferSize: The number of elements the buffer will be able to hold
- bufferType: The type of items that will be used. Can be a ctype or a string, that will then be resolved to a ctype

### Interface.array_to_buffer(array)

Returns a new buffer, and copies the array content to said buffer.
- array: A numpy array or python list that will be copied into the buffer

### Interface.run_function(numThreads, buffers)

Runs the currently set function.
- numThreads: The number of GPU threads that will be started
- buffers: A list of buffers that will be sent to the GPU. The first element of the list will be associated with buffer number 0, the second with 1, etc. If you do not want to associate a buffer with the nth slot, use a None in the list, and continue with the following buffers.

## Buffer

A buffer is a shared part of memory between the GPU and CPU. It is the only way to transfer data to a metal shader.

It can be destroyed using `del buffer`.

The buffer class is private, and any buffer creation should thus be through an interface.

### Buffer.contents()

Returns the contents of the buffer as a numpy array.

### Buffer.release()

Frees up the buffer's memory. Is automatically called on buffer destruction.

### Buffer.interface

The interface the buffer was created from.

## Operators

Operators are available on metal buffers, and will __always__ run on the gpu. If you have a small set of data, or wish to run them on the cpu, use their numpy equivalents on Buffer.contents

Operators available are:
`cos`, `sin`, `sqrt`, `tan`

Note that as of right now, those functions should only be ran on floats or doubles


You can also use inline operations with buffers, as in:
`buffer1 + buffer2` or `buffer1 - buffer2`

Which will also run on the GPU

All of those functions will return a __new buffer__, obtained by applying the operator on every element of the buffer

## Recompiling C libraries.

If you encounter an error regarding a `.dylib` file, or an error that appears to be from the C interface, you need to recompile the C library.

To do this, clone the package's [github repo](https://github.com/Al0den/metalgpu), and go to `metal-gpu-c`. Then, create a copy of the `metal-cpp` folder, that can be found at [this](https://github.com/bkaradzic/metal-cpp) repo. Then, simply run `cmake . && make install`, and it will recompile the library and move it to the correct path.
Then, go to said path (`metalgpu/src/metalgpu/lib/`), and rename from `libmetalgpucpp-arm.dylib` to `libmetalgpucpp-x86.dylib`
