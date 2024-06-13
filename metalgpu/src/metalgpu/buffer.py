import numpy as np

from .utils import anyToMetal, allowedCTypesPointer
from .shader import add_func_kernel, sub_func_kernel, mul_func_kernel, cast_func_kernel

class Buffer:
    def __init__(self, buffPointer : allowedCTypesPointer, buffSize : int, interface, bufNum : int) -> None:
        self.contents : np.ndarray = np.ctypeslib.as_array(buffPointer, shape=(buffSize,))
        self.bufNum = bufNum
        self.interface = interface
        self.bufType = self.contents.dtype

    def release(self) -> None:
        self.interface.release_buffer(self.bufNum)
        self.contents = np.array([])

    def __del__(self) -> None:
        self.release()

    def __add__(self, other : "Buffer") -> "Buffer":
        assert(len(self.contents) == len(other.contents)), "[MetalGPU] Buffers must be of the same content size"
        assert(self.contents.dtype == other.contents.dtype), "[MetalGPU] Buffers must be of the same data type"
        outBuffer = self.interface.create_buffer(len(self.contents), anyToMetal(self.contents.dtype))

        add_kernel = add_func_kernel(self)

        prevShader = self.interface.loaded_shader
        shaderFromPath = self.interface.shader_from_path
        prevFunction = self.interface.current_function
        self.interface.load_shader_from_string(add_kernel)
        self.interface.set_function("add")
        self.interface.run_function(len(self.contents), [self, other, outBuffer])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.interface.set_function(prevFunction)
        return outBuffer

    def __sub__(self, other : "Buffer") -> "Buffer":
        assert(len(self.contents) == len(other.contents)), "[MetalGPU] Buffers must be of the same content size"
        assert(self.contents.dtype == other.contents.dtype), "[MetalGPU] Buffers must be of the same data type"
        outBuffer = self.interface.create_buffer(len(self.contents), anyToMetal(self.contents.dtype))

        sub_kernel = sub_func_kernel(self)

        prevShader = self.interface.loaded_shader
        shaderFromPath = self.interface.shader_from_path
        prevFunction = self.interface.current_function
        self.interface.load_shader_from_string(sub_kernel)
        self.interface.set_function("sub")
        self.interface.run_function(len(self.contents), [self, other, outBuffer])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.interface.set_function(prevFunction)
        return outBuffer

    def __mul__(self, other : "Buffer") -> "Buffer":
        assert(len(self.contents) == len(other.contents)), "[MetalGPU] Buffers must be of the same content size"
        assert(self.contents.dtype == other.contents.dtype), "[MetalGPU] Buffers must be of the same data type"

        outBuffer = self.interface.create_buffer(len(self.contents), anyToMetal(self.contents.dtype))

        mul_kernel = mul_func_kernel(self)

        prevShader = self.interface.loaded_shader
        shaderFromPath = self.interface.shader_from_path
        prevFunction = self.interface.current_function
        self.interface.load_shader_from_string(mul_kernel)
        self.interface.set_function("mul")
        self.interface.run_function(len(self.contents), [self, other, outBuffer])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.interface.set_function(prevFunction)
        return outBuffer

    def astype(self, targetType) -> "Buffer":
        new_buf = self.interface.create_buffer(len(self.contents), targetType)

        cast_kernel = cast_func_kernel(self, targetType)

        prevShader = self.interface.loaded_shader
        shaderFromPath = self.interface.shader_from_path
        prevFunction = self.interface.current_function
        self.interface.load_shader_from_string(cast_kernel)

        self.interface.set_function("cast")

        self.interface.run_function(len(self.contents), [self, new_buf])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.interface.set_function(prevFunction)
        return new_buf
