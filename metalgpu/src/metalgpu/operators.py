import numpy as np

from .buffer import Buffer
from .utils import anyToMetal
from .shader import sqrt_func_kernel, cos_func_kernel, sin_func_kernel, tan_func_kernel

def sqrt(buf : Buffer) -> "Buffer":
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    sqrt_kernel = sqrt_func_kernel(buf)

    
    prevShader = buf.interface.loaded_shader
    shaderFromPath = buf.interface.shader_from_path
    buf.interface.load_shader_from_string(sqrt_kernel)
    buf.interface.set_function("sqrt_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer


def cos(buf : Buffer) -> "Buffer":
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    cos_kernel = cos_func_kernel(buf)

    prevShader = buf.interface.loaded_shader
    shaderFromPath = buf.interface.shader_from_path
    buf.interface.load_shader_from_string(cos_kernel)
    buf.interface.set_function("cos_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer

def sin(buf : Buffer) -> "Buffer":
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    sin_kernel = sin_func_kernel(buf)

    prevShader = buf.interface.loaded_shader
    shaderFromPath = buf.interface.shader_from_path
    buf.interface.load_shader_from_string(sin_kernel)
    buf.interface.set_function("sin_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer


def tan(buf : Buffer) -> "Buffer":
    if(buf.contents.dtype != np.float32 and buf.contents.dtype != np.float64): raise TypeError("[MetalGPU] Buffer data type must be float or double")
    out_buffer = buf.interface.create_buffer(len(buf.contents), anyToMetal(buf.contents.dtype))
    tan_kernel = tan_func_kernel(buf)

    prevShader = buf.interface.loaded_shader
    shaderFromPath = buf.interface.shader_from_path
    buf.interface.load_shader_from_string(tan_kernel)
    buf.interface.set_function("tan_func")
    buf.interface.run_function(len(buf.contents), [buf, out_buffer])
    buf.interface.load_shader(prevShader) if shaderFromPath else buf.interface.load_shader_from_string(prevShader)
    return out_buffer