import numpy as np
import ctypes

# table[0]: Metal type, table[1]: numpy type, table[2]: ctypes type, table[2:]: Unspecified
tables = [
    ["int", np.int32, ctypes.c_int],
    ["float", np.float32, ctypes.c_float],
    ["half", np.int16, ctypes.c_int16, "short"],
    ["bool", np.bool_, ctypes.c_bool, np.byte],
    ["long", np.int64, ctypes.c_long, "int64", "int64_t"],
    ["uint64_t", np.uint64, ctypes.c_ulong, "uint64", "unsigned long"],
    ["uint32_t", np.uint32, ctypes.c_uint, "uint32", "uint", "unsigned int"],
    ["uint16_t", np.uint16, ctypes.c_uint16, "uint16", "ushort", "unsigned short"]
]

allowedCTypes = ctypes.c_int | ctypes.c_float | ctypes.c_int16 | ctypes.c_bool | ctypes.c_long | ctypes.c_ulong | ctypes.c_uint | ctypes.c_uint16 | ctypes.c_uint32 | ctypes.c_uint64
allowedCTypesPointer = ctypes.POINTER(ctypes.c_int) | ctypes.POINTER(ctypes.c_float) | ctypes.POINTER(ctypes.c_int16) | ctypes.POINTER(ctypes.c_bool) | ctypes.POINTER(ctypes.c_long) | ctypes.POINTER(ctypes.c_ulong) | ctypes.POINTER(ctypes.c_uint) | ctypes.POINTER(ctypes.c_uint16) | ctypes.POINTER(ctypes.c_uint32) | ctypes.POINTER(ctypes.c_uint64)
allowedNumpyTypes = np.int32 | np.float32 | np.int16 | np.bool_ | np.int64 | np.uint64 | np.uint32 | np.uint16

def anyToMetal(receivedType : str | allowedNumpyTypes | allowedCTypes) -> str:
    for table in tables:
        if receivedType in table:
            return table[0]
    raise TypeError("[MetalGPU] Unsupported data type")


def anyToCtypes(receivedType : str | allowedNumpyTypes | allowedCTypes) -> allowedCTypes:
    for table in tables:
        if receivedType in table:
            return table[2]
    raise TypeError("[MetalGPU] Unsupported data type")


def anyToNumpy(receivedType : str | allowedNumpyTypes | allowedCTypes) -> allowedNumpyTypes:
    for table in tables:
        if receivedType in table:
            return table[1]
    raise TypeError("[MetalGPU] Unsupported data type")
