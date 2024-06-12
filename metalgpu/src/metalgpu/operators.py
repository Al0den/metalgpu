import numpy as np

from .buffer import Buffer
from .utils import anyToMetal

def sqrt(buf : Buffer):
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    sqrt_kernel = f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void sqrt_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = sqrt(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(sqrt_kernel)
    buf.interface.set_function("sqrt_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer


def cos(buf : Buffer):
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    cos_kernel = f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void cos_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = cos(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(cos_kernel)
    buf.interface.set_function("cos_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer

def sin(buf : Buffer):
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    sin_kernel = f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void sin_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = sin(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(sin_kernel)
    buf.interface.set_function("sin_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer


def tan(buf : Buffer):
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    tan_kernel = f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void tan_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = tan(arr1[id]);
    }};
    """
    prevShader = buf.interface._loaded_shader
    shaderFromPath = buf.interface._shader_from_path
    buf.interface.load_shader_from_string(tan_kernel)
    buf.interface.set_function("tan_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer
