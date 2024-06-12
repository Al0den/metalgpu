import ctypes
import numpy as np

import os

from .buffer import Buffer
from .utils import anyToCtypes, anyToMetal, allowedCTypes

class Interface:
    def __init__(self):
        _objPath = os.path.dirname(__file__)

        assert os.path.isfile(_objPath + "/lib/libmetalgpucpp-arm.dylib"), "[MetalGPU] Library not found, please run `python -m metalgpu build`"
        self._metal = ctypes.cdll.LoadLibrary(_objPath + "/lib/libmetalgpucpp-arm.dylib")

        self._init_functions()
        self._init()
        self._loaded_shader = """
        #include <metal_stdlib>
        
        using namespace metal;

        kernel void emptyFunc() {};
        """
        self._shader_from_path = False
        self.current_function = "emptyFunc"

        self.load_shader_from_string(self._loaded_shader)
        self.set_function("emptyFunc")

    def __del__(self):
        self._deleteInstance()

    def _init_functions(self):
        self._init = self._metal.init
        self._createBuffer = self._metal.createBuffer
        self._createLibrary = self._metal.createLibrary
        self._setFunction = self._metal.setFunction
        self._runFunction = self._metal.runFunction
        self._releaseBuffer = self._metal.releaseBuffer
        self._getBufferPointer = self._metal.getBufferPointer
        self._deleteInstance = self._metal.deleteInstance
        self._createLibraryFromString = self._metal.createLibraryFromString

        self._init.argtypes = []
        self._createBuffer.argtypes = [ctypes.c_int]
        self._createLibrary.argtypes = [ctypes.c_char_p]
        self._setFunction.argtypes = [ctypes.c_char_p]
        self._runFunction.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._releaseBuffer.argtypes = [ctypes.c_int]
        self._getBufferPointer.argtypes = [ctypes.c_int]
        self._deleteInstance.argtypes = []
        self._createLibraryFromString.argtypes = [ctypes.c_char_p]

        self._init.restype = None
        self._createBuffer.restype = ctypes.c_int
        self._createLibrary.restype = None
        self._setFunction.restype = None
        self._runFunction.restype = None
        self._releaseBuffer.restype = None
        self._getBufferPointer.restype = ctypes.POINTER(ctypes.c_int)
        self._deleteInstance.restype = None
        self._createLibraryFromString.argtypes = [ctypes.c_char_p]

    def create_buffer(self, bufsize : int, bufferTypeRaw : str | np.ndarray | allowedCTypes):
        assert bufsize > 0, "[MetalGPU] Buffer size must be greater than 0"
        bufType = anyToCtypes(bufferTypeRaw)
        number = self._createBuffer(ctypes.sizeof(bufType) * bufsize)
        self._getBufferPointer.restype = ctypes.POINTER(bufType)
        buffPointer = self._getBufferPointer(number)
        buff = Buffer(buffPointer, bufsize, self, number)
        return buff

    def load_shader(self, shaderPath : str):
        self._createLibrary(shaderPath.encode('utf-8'))
        self._loaded_shader = shaderPath
        self._shader_from_path = True

    def set_function(self, functionName : str):
        self._setFunction(functionName.encode('utf-8'))
        self.current_function = functionName

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

    def array_to_buffer(self, array : np.ndarray | list):
        array = np.array(array)
        if array.ndim != 1: raise Exception("[MetalGPU] Array must be 1D")

        buftype = anyToMetal(array.dtype)
    
        buffer = self.create_buffer(len(array), buftype)

        buffer.contents[:] = array
        return buffer

    def load_shader_from_string(self, libStr : str):
        self._createLibraryFromString(libStr.encode('utf-8'))
        self._loaded_shader = libStr
        self._shader_from_path = False
