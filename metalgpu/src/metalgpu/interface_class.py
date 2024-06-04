import ctypes
import numpy as np
import os

class Buffer:
    def __init__(self, buffPointer, buffSize, buffNum, interface):
        self.contents = np.ctypeslib.as_array(buffPointer, shape=(buffSize,))
        self.bufnum = buffNum
        self.interface = interface

    def release(self):
        self.interface.releaseBuffer(self.bufnum)
        self.contents = []

class Interface:
    def __init__(self):
        _objPath = os.path.dirname(__file__)
        self._metal = ctypes.cdll.LoadLibrary(_objPath + "/binaries/lib.so")
        self._init_functions()
        self._init()

    def _init_functions(self):
        self._init = self._metal.init
        self._createBuffer = self._metal.createBuffer
        self._createLibrary = self._metal.createLibrary
        self._setFunction = self._metal.setFunction
        self._runFunction = self._metal.runFunction
        self._releaseBuffer = self._metal.releaseBuffer

        self._init.argtypes = []
        self._createBuffer.argtypes = [ctypes.c_int, ctypes.c_int]
        self._createLibrary.argtypes = [ctypes.c_char_p]
        self._setFunction.argtypes = [ctypes.c_char_p]
        self._runFunction.argtypes = [ctypes.c_int]
        self._releaseBuffer.argtypes = [ctypes.c_int]

        self._init.restype = None
        self._createBuffer.restype = ctypes.POINTER(ctypes.c_int)
        self._createLibrary.restype = None
        self._setFunction.restype = None
        self._runFunction.restype = None
        self._releaseBuffer.restype = None

    def create_buffer(self, bufsize : int, bufnum : int, bufType):
        self._createBuffer.restype = ctypes.POINTER(bufType)
        buffPointer = self._createBuffer(ctypes.sizeof(bufType) * bufsize, bufnum)
        buff = Buffer(buffPointer, bufsize, bufnum, self)
        return buff

    def load_shader(self, shaderPath):
        self._createLibrary(shaderPath.encode('utf-8'))

    def set_function(self, functionName):
        self._setFunction(functionName.encode('utf-8'))

    def run_function(self, numThreads):
        self._runFunction(numThreads)

    def release_buffer(self, bufnum):
        self._releaseBuffer(bufnum)

    def array_to_buffer(self, array, bufNum):
        type = array.dtype
        if type == np.int32: type = ctypes.c_int
        elif type == np.float32: type = ctypes.c_float
        elif type == np.float64: type = ctypes.c_double
        elif type == np.int64: type = ctypes.c_longlong
        elif type == np.int16: type = ctypes.c_short
        elif type == np.int8: type = ctypes.c_byte
        elif type == np.uint8: type = ctypes.c_ubyte
        elif type == np.uint16: type = ctypes.c_ushort
        elif type == np.uint32: type = ctypes.c_uint
        elif type == np.uint64: type = ctypes.c_ulonglong
        else: raise Exception("Unsupported data type")

        buffer = self.createBuffer(len(array), bufNum, type)

        buffer.contents[:] = array
        return buffer



        
