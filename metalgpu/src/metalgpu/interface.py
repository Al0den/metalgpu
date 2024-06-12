import ctypes
import numpy as np

import os

from .buffer import Buffer
from .utils import anyToCtypes, anyToMetal, allowedCTypes, allowedNumpyTypes

class MetalSize:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

class Interface:
    def __init__(self) -> None:
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

    def __del__(self) -> None:
        self._deleteInstance()

    def _init_functions(self) -> None:
        self._init = self._metal.init
        self._createBuffer = self._metal.createBuffer
        self._createLibrary = self._metal.createLibrary
        self._setFunction = self._metal.setFunction
        self._runFunction = self._metal.runFunction
        self._releaseBuffer = self._metal.releaseBuffer
        self._getBufferPointer = self._metal.getBufferPointer
        self._deleteInstance = self._metal.deleteInstance
        self._createLibraryFromString = self._metal.createLibraryFromString
        self._maxThreadsPerGroup = self._metal.maxThreadsPerGroup
        self._threadExecutionWidth = self._metal.threadExecutionWidth

        self._init.argtypes = []
        self._createBuffer.argtypes = [ctypes.c_int]
        self._createLibrary.argtypes = [ctypes.c_char_p]
        self._setFunction.argtypes = [ctypes.c_char_p]
        self._runFunction.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._releaseBuffer.argtypes = [ctypes.c_int]
        self._getBufferPointer.argtypes = [ctypes.c_int]
        self._deleteInstance.argtypes = []
        self._createLibraryFromString.argtypes = [ctypes.c_char_p]
        self._maxThreadsPerGroup.argtypes = []
        self._threadExecutionWidth.argtypes = []

        self._init.restype = None
        self._createBuffer.restype = ctypes.c_int
        self._createLibrary.restype = None
        self._setFunction.restype = None
        self._runFunction.restype = None
        self._releaseBuffer.restype = None
        self._getBufferPointer.restype = ctypes.POINTER(ctypes.c_int)
        self._deleteInstance.restype = None
        self._createLibraryFromString.restype = None
        self._maxThreadsPerGroup.restype = int
        self._threadExecutionWidth.restype = int

    def create_buffer(self, bufsize: int, buffer_type: str | allowedNumpyTypes | allowedCTypes) -> "Buffer":
        assert bufsize > 0, "[MetalGPU] Buffer size must be greater than 0"
        bufType = anyToCtypes(buffer_type)
        number = self._createBuffer(ctypes.sizeof(bufType) * bufsize)
        self._getBufferPointer.restype = ctypes.POINTER(bufType)
        buffPointer = self._getBufferPointer(number)
        buff = Buffer(buffPointer, bufsize, self, number)
        return buff

    def load_shader(self, shader_path: str) -> None:
        self._createLibrary(shader_path.encode('utf-8'))
        self._loaded_shader = shader_path
        self._shader_from_path = True

    def set_function(self, function_name: str) -> None:
        self._setFunction(function_name.encode('utf-8'))
        self.current_function = function_name

    def run_function(self, received_size: int | MetalSize, buffers: list[int], function_name: str | None = None) -> None:
        if isinstance(received_size, int):
            received_size = MetalSize(received_size, 1, 1)

        if function_name is not None:
            self.set_function(function_name)

        bufferList = []
        for buff in buffers:
            if buff is None:
                bufferList.append(-1)
            elif isinstance(buff, Buffer):
                bufferList.append(buff.bufNum)
            else:
                raise Exception("Unsupported buffer type")

        bufferArr = np.array(bufferList).astype(np.int32)
        metalSize = np.array([received_size.width, received_size.height, received_size.depth]).astype(np.int32)

        metalSizePointer = metalSize.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        bufferPointer = bufferArr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        self._runFunction(metalSizePointer, bufferPointer, len(bufferArr))

    def release_buffer(self, bufnum: int) -> None:
        self._releaseBuffer(bufnum)

    def array_to_buffer(self, array: np.ndarray | list) -> "Buffer":
        array = np.array(array)
        if array.ndim != 1:
            array = array.flatten()

        buftype = anyToMetal(array.dtype)

        buffer = self.create_buffer(len(array), buftype)

        buffer.contents[:] = array
        return buffer

    def load_shader_from_string(self, shader_string: str) -> None:
        self._createLibraryFromString(shader_string.encode('utf-8'))
        self._loaded_shader = shader_string
        self._shader_from_path = False

    def threadExecutionWidth(self):
        return self._threadExecutionWidth()

    def maxThreadsPerGroup(self):
        return self._maxThreadsPerGroup()
