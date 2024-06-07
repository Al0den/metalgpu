import numpy as np

class Buffer:
    def __init__(self, buffPointer, buffSize, interface, bufNum):
        self.contents = np.ctypeslib.as_array(buffPointer, shape=(buffSize,))
        self.bufNum = bufNum
        self.interface = interface
        self.bufType = self.contents.dtype

    def release(self):
        self.interface.release_buffer(self.bufNum)
        self.contents = np.array([])

    def __del__(self):
        self.release()

    def __add__(self, other):
        assert(other.__class__ == Buffer), "[MetalGPU] Adding a buffer to a non-buffer is not supported"
        assert(len(self.contents) == len(other.contents)), "[MetalGPU] Buffers must be of the same content size"
        assert(self.contents.dtype == other.contents.dtype), "[MetalGPU] Buffers must be of the same data type"
        outBuffer = self.interface.create_buffer(len(self.contents), self.toMetalType(self.contents.dtype))

        add_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void add(const device {self.toMetalType(self.contents.dtype)} *arr1 [[buffer(0)]], const device {self.toMetalType(self.contents.dtype)} *arr2 [[buffer(1)]], device {self.toMetalType(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
            arr3[id] = arr1[id] + arr2[id];
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
        self.interface.load_shader_from_string(add_kernel)
        self.interface.set_function("add")
        self.interface.run_function(len(self.contents), [self, other, outBuffer])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        return outBuffer

    def __sub__(self, other):
        assert(other.__class__ == Buffer), "[MetalGPU] Subtracting a buffer from a non-buffer is not supported"
        assert(len(self.contents) == len(other.contents)), "[MetalGPU] Buffers must be of the same content size"
        assert(self.contents.dtype == other.contents.dtype), "[MetalGPU] Buffers must be of the same data type"
        outBuffer = self.interface.create_buffer(len(self.contents), self.toMetalType(self.contents.dtype))
        sub_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void sub(const device {self.toMetalType(self.contents.dtype)} *arr1 [[buffer(0)]], const device {self.toMetalType(self.contents.dtype)} *arr2 [[buffer(1)]], device {self.toMetalType(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
            arr3[id] = arr1[id] - arr2[id];
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
        self.interface.load_shader_from_string(sub_kernel)
        self.interface.set_function("sub")
        self.interface.run_function(len(self.contents), [self, other, outBuffer])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        return outBuffer

    def __mul__(self, other):
        assert(other.__class__ == Buffer), "[MetalGPU] Multiplying a buffer with a non-buffer is not supported"
        assert(len(self.contents) == len(other.contents)), "[MetalGPU] Buffers must be of the same content size"
        assert(self.contents.dtype == other.contents.dtype), "[MetalGPU] Buffers must be of the same data type"

        outBuffer = self.interface.create_buffer(len(self.contents), self.toMetalType(self.contents.dtype))
        mul_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void mul(const device {self.toMetalType(self.contents.dtype)} *arr1 [[buffer(0)]], const device {self.toMetalType(self.contents.dtype)} *arr2 [[buffer(1)]], device {self.toMetalType(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
            arr3[id] = arr1[id] * arr2[id];
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
        self.interface.load_shader_from_string(mul_kernel)
        self.interface.run_function(len(self.contents), [self, other, outBuffer])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        return outBuffer


    def as_float(self):
        new_buf = self.interface.create_buffer(len(self.contents), "float")
        cast_kernel = f"""
        #include <metal_stdlib>

        using namespace metal;

        kernel void cast(const device {self.toMetalType(self.contents.dtype)} *arr1 [[buffer(0)]], device float *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
            arr2[id] = float(arr1[id]);
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
        self.interface.load_shader_from_string(cast_kernel)
        self.interface.set_function("cast")
        self.interface.run_function(len(self.contents), [self, new_buf])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.release()
        self.contents = new_buf.contents
        self.bufType = self.contents.dtype
        return

    def as_int(self):
        new_buf = self.interface.create_buffer(len(self.contents), "int")
        cast_kernel = f"""
        #include <metal_stdlib>
        using namespace metal;
        kernel void cast(const device {self.toMetalType(self.contents.dtype)} *arr1 [[buffer(0)]], device int *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
            arr2[id] = int(arr1[id]);
        }};
        """
        prevShader = self.interface._loaded_shader
        shaderFromPath = self.interface._shader_from_path
        self.interface.load_shader_from_string(cast_kernel)
        self.interface.set_function("cast")
        self.interface.run_function(len(self.contents), [self, new_buf])
        self.interface.load_shader(prevShader) if shaderFromPath else self.interface.load_shader_from_string(prevShader)
        self.release()
        self.contents = new_buf.contents
        self.bufType = self.contents.dtype
        return

    def toMetalType(self, numpyType):
        if numpyType == np.int32: return "int"
        elif numpyType == np.float32: return "float"
        elif numpyType == np.int16: return "half"
        else : raise Exception("[MetalGPU] Type not supported, convert to int32/float")


