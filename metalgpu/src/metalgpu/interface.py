import ctypes
import numpy as np

import os

from .buffer import Buffer
from .utils import anyToCtypes, anyToMetal, allowedCTypes, allowedNumpyTypes
from .shader import initial_shader


class MetalSize:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth


class Interface:
    def __init__(self) -> None:
        _objPath = os.path.dirname(__file__)

        assert os.path.isfile(_objPath + "/lib/libmetalgpucpp-arm.dylib"), "[MetalGPU] Library not found, please run `python -m metalgpu build`"
        self.__metal = ctypes.cdll.LoadLibrary(_objPath + "/lib/libmetalgpucpp-arm.dylib")

        self.__init_functions()
        self.__init()

        self.load_shader_from_string(initial_shader())
        self.set_function("emptyFunc")

    def __del__(self) -> None:
        self.__deleteInstance()

    def __init_functions(self) -> None:
        self.__init = self.__metal.init
        self.__createBuffer = self.__metal.createBuffer
        self.__createLibrary = self.__metal.createLibrary
        self.__setFunction = self.__metal.setFunction
        self.__runFunction = self.__metal.runFunction
        self.__releaseBuffer = self.__metal.releaseBuffer
        self.__getBufferPointer = self.__metal.getBufferPointer
        self.__deleteInstance = self.__metal.deleteInstance
        self.__createLibraryFromString = self.__metal.createLibraryFromString
        self.__maxThreadsPerGroup = self.__metal.maxThreadsPerGroup
        self.__threadExecutionWidth = self.__metal.threadExecutionWidth

        self.__init.argtypes = []
        self.__createBuffer.argtypes = [ctypes.c_int]
        self.__createLibrary.argtypes = [ctypes.c_char_p]
        self.__setFunction.argtypes = [ctypes.c_char_p]
        self.__runFunction.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool]
        self.__releaseBuffer.argtypes = [ctypes.c_int]
        self.__getBufferPointer.argtypes = [ctypes.c_int]
        self.__deleteInstance.argtypes = []
        self.__createLibraryFromString.argtypes = [ctypes.c_char_p]
        self.__maxThreadsPerGroup.argtypes = []
        self.__threadExecutionWidth.argtypes = []

        self.__init.restype = None
        self.__createBuffer.restype = ctypes.c_int
        self.__createLibrary.restype = None
        self.__setFunction.restype = None
        self.__runFunction.restype = None
        self.__releaseBuffer.restype = None
        self.__getBufferPointer.restype = ctypes.POINTER(ctypes.c_int)
        self.__deleteInstance.restype = None
        self.__createLibraryFromString.restype = None
        self.__maxThreadsPerGroup.restype = int
        self.__threadExecutionWidth.restype = int

    def create_buffer(self, bufsize: int, buffer_type: str | allowedNumpyTypes | allowedCTypes) -> "Buffer":
        assert bufsize > 0, "[MetalGPU] Buffer size must be greater than 0"
        bufType = anyToCtypes(buffer_type)
        number = self.__createBuffer(ctypes.sizeof(bufType) * bufsize)
        self.__getBufferPointer.restype = ctypes.POINTER(bufType)
        buffPointer = self.__getBufferPointer(number)
        buff = Buffer(buffPointer, bufsize, self, number)
        return buff

    def load_shader(self, shader_path: str) -> None:
        self.__createLibrary(shader_path.encode('utf-8'))
        self.loaded_shader = shader_path
        self.shader_from_path = True

    def set_function(self, function_name: str) -> None:
        self.__setFunction(function_name.encode('utf-8'))
        self.current_function = function_name

    def run_function(self, received_size: int | MetalSize, buffers: list[Buffer], function_name: str | None = None, wait_for_completion : bool = True) -> None:
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

        self.__runFunction(metalSizePointer, bufferPointer, len(bufferArr), wait_for_completion)

    def release_buffer(self, bufnum: int) -> None:
        self.__releaseBuffer(bufnum)

    def array_to_buffer(self, array: np.ndarray | list) -> "Buffer":
        array = np.array(array)
        if array.ndim != 1:
            array = array.flatten()

        buftype = anyToMetal(array.dtype)

        buffer = self.create_buffer(len(array), buftype)

        buffer.contents[:] = array
        return buffer

    def load_shader_from_string(self, shader_string: str) -> None:
        self.__createLibraryFromString(shader_string.encode('utf-8'))
        self.loaded_shader = shader_string
        self.shader_from_path = False

    def threadExecutionWidth(self):
        return self.__threadExecutionWidth()

    def maxThreadsPerGroup(self):
        return self.__maxThreadsPerGroup()
