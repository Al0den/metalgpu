from .utils import anyToMetal


def add_func_kernel(self):
    return f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void add(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], const device {anyToMetal(self.contents.dtype)} *arr2 [[buffer(1)]], device {anyToMetal(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
        arr3[id] = arr1[id] + arr2[id];
    }};
    """

def sub_func_kernel(self):
    return f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void sub(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], const device {anyToMetal(self.contents.dtype)} *arr2 [[buffer(1)]], device {anyToMetal(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
        arr3[id] = arr1[id] - arr2[id];
    }};
    """

def mul_func_kernel(self):
    return f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void mul(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], const device {anyToMetal(self.contents.dtype)} *arr2 [[buffer(1)]], device {anyToMetal(self.contents.dtype)} *arr3 [[buffer(2)]], uint id [[thread_position_in_grid]]) {{
        arr3[id] = arr2[id] * arr1[id];
    }};
    """

def cast_func_kernel(self, targetType):
    return f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void cast(const device {anyToMetal(self.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(targetType)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = float(arr1[id]);
    }};
    """

def initial_shader():
    return """
    #include <metal_stdlib>

    using namespace metal;

    kernel void emptyFunc() {};
    """

def sqrt_func_kernel(buf):
    return f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void sqrt_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = sqrt(arr1[id]);
    }};
    """

def cos_func_kernel(buf):
    return f"""
    #include <metal_stdlib>

    using namespace metal;

    kernel void cos_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = cos(arr1[id]);
    }};
    """

def sin_func_kernel(buf):
    return f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void sin_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = sin(arr1[id]);
    }};
    """

def tan_func_kernel(buf):
    return f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void tan_func(device {anyToMetal(buf.contents.dtype)} *arr1 [[buffer(0)]], device {anyToMetal(buf.contents.dtype)} *arr2 [[buffer(1)]], uint id [[thread_position_in_grid]]) {{
        arr2[id] = tan(arr1[id]);
    }};
    """