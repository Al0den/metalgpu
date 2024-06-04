import ctypes
def getCtypeType(userInput):
    if userInput == "int":
        return ctypes.c_int
    elif userInput == "float":
        return ctypes.c_float
    elif userInput == "double":
        return ctypes.c_double
    elif userInput == "longlong":
        return ctypes.c_longlong
    elif userInput == "char":
        return ctypes.c_char
    elif userInput == "short":
        return ctypes.c_short
    elif userInput == "long":
        return ctypes.c_long
    elif userInput == "longdouble":
        return ctypes.c_longdouble
    elif userInput == "int8":
        return ctypes.c_int8
    elif userInput == "int16":
        return ctypes.c_int16
    elif userInput == "int32":
        return ctypes.c_int32
    elif userInput == "int64":
        return ctypes.c_int64
    elif userInput == "uint8":
        return ctypes.c_uint8
    elif userInput == "uint16":
        return ctypes.c_uint16
    elif userInput == "uint32":
        return ctypes.c_uint32
    elif userInput == "uint64":
        return ctypes.c_uint64
    elif userInput == "ubyte":
        return ctypes.c_ubyte
    elif userInput == "ushort":
        return ctypes.c_ushort
    elif userInput == "uint":
        return ctypes.c_uint
    elif userInput == "ulong":
        return ctypes.c_ulong
    elif userInput == "ulonglong":
        return ctypes.c_ulonglong

