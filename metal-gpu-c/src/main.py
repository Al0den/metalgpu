import ctypes
import numpy as np

shader = ctypes.cdll.LoadLibrary('./build/lib.so')

init = shader.init
init.argtypes = []
init.restype = None

createBuffer = shader.createBuffer
createBuffer.argtypes = [ctypes.c_int]
createBuffer.restype = ctypes.c_int

createLibrary = shader.createLibrary
createLibrary.argtypes = [ctypes.c_char_p]
createLibrary.restype = None

createLibraryFromString = shader.createLibraryFromString
createLibraryFromString.argtypes = [ctypes.c_char_p]
createLibraryFromString.restype = None

setFunction = shader.setFunction
setFunction.argtypes = [ctypes.c_char_p]
setFunction.restype = None

runFunction = shader.runFunction
runFunction.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
runFunction.restype = None

getBufferPointer = shader.getBufferPointer
getBufferPointer.argtypes = [ctypes.c_int]
getBufferPointer.restype = ctypes.POINTER(ctypes.c_int)

libStr = """
#include <metal_stdlib>

using namespace metal;

void kernel cos_func(device float* in [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    in[id] = cos(in[id]);
};

"""

init()
createLibraryFromString(libStr.encode('utf-8'))
setFunction("cos_func".encode('utf-8'))

arr_size = 10

buffNumber1 = createBuffer(ctypes.sizeof(ctypes.c_float) * arr_size)
arr1 = np.ctypeslib.as_array(getBufferPointer(buffNumber1), shape=(arr_size,))

for i in range(arr_size):
    arr1[i] = i/arr_size

buffers = np.array([0])

runFunction(arr_size, buffers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
print(arr1)

