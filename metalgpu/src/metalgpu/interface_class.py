import ctypes
import numpy as np
import os


def createInterface():
    return Interface()


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
        self._initFunctions()
        self._init()

    def _initFunctions(self):
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

    def createBuffer(self, bufsize, bufnum, bufType):
        self._createBuffer.restype = ctypes.POINTER(bufType)
        buffPointer = self._createBuffer(ctypes.sizeof(bufType) * bufsize, bufnum)
        buff = Buffer(buffPointer, bufsize, bufnum, self)
        return buff

    def loadShader(self, shaderPath):
        self._createLibrary(shaderPath.encode('utf-8'))

    def setFunction(self, functionName):
        self._setFunction(functionName.encode('utf-8'))

    def runFunction(self, numThreads):
        self._runFunction(numThreads)

    def releaseBuffer(self, bufnum):
        self._releaseBuffer(bufnum)
