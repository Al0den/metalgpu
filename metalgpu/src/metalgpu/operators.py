from interface_class import Buffer
import numpy as np


def sqrt(buf):
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")

    sqrt_kernel = f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void sqrt_func(device {buf.toMetalType(buf.contents.dtype)} *arr1 [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
        arr1[id] = sqrt(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(sqrt_kernel)
    buf.interface.set_function("sqrt_func")
    buf.interface.run_function(len(buf.contents), [buf])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)


def cos(buf):
    # If type is np.ndarray, check if it is float32 or float64
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")

    cos_kernel = f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void cos_func(device {buf.toMetalType(buf.contents.dtype)} *arr1 [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
        arr1[id] = cos(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(cos_kernel)
    buf.interface.set_function("cos_func")
    buf.interface.run_function(len(buf.contents), [buf])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)


def sin(buf):
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")
    sin_kernel = f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void sin_func(device {buf.toMetalType(buf.contents.dtype)} *arr1 [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
        arr1[id] = sin(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(sin_kernel)
    buf.interface.set_function("sin_func")
    buf.interface.run_function(len(buf.contents), [buf])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)


def tan(buf):
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")
    tan_kernel = f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void tan_func(device {buf.toMetalType(buf.contents.dtype)} *arr1 [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
        arr1[id] = tan(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(tan_kernel)
    buf.interface.set_function("tan_func")
    buf.interface.run_function(len(buf.contents), [buf])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
