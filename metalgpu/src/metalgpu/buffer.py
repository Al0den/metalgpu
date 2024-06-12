import numpy as np
from .utils import anyToMetal, allowedCTypesPointer

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

        add_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void add(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], const device {anyToMetal(self.contents.dtype)} *arr2 [[buffer(1)]], device {anyToMetal(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
            arr3[id] = arr1[id] + arr2[id];
        }};
        """

        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
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
        sub_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void sub(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], const device {anyToMetal(self.contents.dtype)} *arr2 [[buffer(1)]], device {anyToMetal(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
            arr3[id] = arr1[id] - arr2[id];
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
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
        mul_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void mul(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], const device {anyToMetal(self.contents.dtype)} *arr2 [[buffer(1)]], device {anyToMetal(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
            arr3[id] = arr2[id] * arr1[id];
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
        prevFunction = self.interface.current_function
        self.interface.load_shader_from_string(mul_kernel)
        self.interface.set_function("mul")
        self.interface.run_function(len(self.contents), [self, other, outBuffer])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.interface.set_function(prevFunction)
        return outBuffer

    def astype(self, targetType) -> "Buffer":
        new_buf = self.interface.create_buffer(len(self.contents), targetType)
        cast_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void cast(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(targetType)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
            arr2[id] = float(arr1[id]);
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
        prevFunction = self.interface.current_function
        self.interface.load_shader_from_string(cast_kernel)
        self.interface.set_function("cast")
        self.interface.run_function(len(self.contents), [self, new_buf])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.interface.set_function(prevFunction)
        return new_buf
