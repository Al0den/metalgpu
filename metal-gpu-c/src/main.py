import ctypes
import numpy as np

shader = ctypes.cdll.LoadLibrary('./build/lib.so')

init = shader.init
init.argtypes = []
init.restype = None

createBuffer = shader.createBuffer
createBuffer.argtypes = [ctypes.c_int, ctypes.c_int64]
createBuffer.restype = ctypes.POINTER(ctypes.c_int)

createLibrary = shader.createLibrary
createLibrary.argtypes = [ctypes.c_char_p]
createLibrary.restype = None

setFunction = shader.setFunction
setFunction.argtypes = [ctypes.c_char_p]
setFunction.restype = None

runFunction = shader.runFunction
runFunction.argtypes = [ctypes.c_int]
runFunction.restype = None

init()
createLibrary("./src/test.metal".encode('utf-8'))
setFunction("adder".encode('utf-8'))

arr_size = 1000

buffPointer1 = createBuffer(ctypes.sizeof(ctypes.c_int) * arr_size, 0)
arr1 = np.ctypeslib.as_array(buffPointer1, shape=(arr_size,))

buffPointer2 = createBuffer(ctypes.sizeof(ctypes.c_int) * arr_size, 1)
arr2 = np.ctypeslib.as_array(buffPointer2, shape=(arr_size,))

buffPointer3 = createBuffer(ctypes.sizeof(ctypes.c_int) * arr_size, 2)
arr3 = np.ctypeslib.as_array(buffPointer3, shape=(arr_size,))

for i in range(arr_size):
    arr1[i] = i
    arr2[i] = 5

runFunction(arr_size)

print(arr1[100], arr2[100], arr3[100])

