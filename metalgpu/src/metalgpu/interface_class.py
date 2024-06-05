import ctypes
import numpy as np
import os

class Buffer:
    def __init__(self, buffPointer, buffSize, interface, bufNum):
        self.contents = np.ctypeslib.as_array(buffPointer, shape=(buffSize,))
        self.bufNum = bufNum
        self.interface = interface

    def release(self):
        self.interface.release_buffer(self.bufNum)
        self.contents = []

    def __del__(self):
        self.release()

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
        self._getBufferPointer = self._metal.getBufferPointer
        self._deleteInstance = self._metal.deleteInstance

        self._init.argtypes = []
        self._createBuffer.argtypes = [ctypes.c_int]
        self._createLibrary.argtypes = [ctypes.c_char_p]
        self._setFunction.argtypes = [ctypes.c_char_p]
        self._runFunction.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._releaseBuffer.argtypes = [ctypes.c_int]
        self._getBufferPointer.argtypes = [ctypes.c_int]
        self._deleteInstance.argtypes = []

        self._init.restype = None
        self._createBuffer.restype = ctypes.c_int
        self._createLibrary.restype = None
        self._setFunction.restype = None
        self._runFunction.restype = None
        self._releaseBuffer.restype = None
        self._getBufferPointer.restype = ctypes.POINTER(ctypes.c_int)
        self._deleteInstance.restype = None

    def create_buffer(self, bufsize : int, bufType):
        number = self._createBuffer(ctypes.sizeof(bufType) * bufsize)
        self._getBufferPointer.restype = ctypes.POINTER(bufType)
        buffPointer = self._getBufferPointer(number)
        buff = Buffer(buffPointer, bufsize, self, number)
        return buff

    def load_shader(self, shaderPath : str):
        self._createLibrary(shaderPath.encode('utf-8'))

    def set_function(self, functionName : str):
        self._setFunction(functionName.encode('utf-8'))

    def run_function(self, numThreads : int, buffers : list):
        bufferList = []
        for buff in buffers:
            if buff is None:
                bufferList.append(-1)
            elif isinstance(buff, Buffer):
                bufferList.append(buff.bufNum)
            else:
                raise Exception("Unsupported buffer type")
        bufferArr = np.array(bufferList).astype(np.int32)
        self._runFunction(numThreads, bufferArr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(bufferArr))

    def release_buffer(self, bufnum : int):
        self._releaseBuffer(bufnum)

    def array_to_buffer(self, array):
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

        buffer = self.create_buffer(len(array), type)

        buffer.contents[:] = array
        return buffer

    def __del__(self):
        self._deleteInstance()


        
