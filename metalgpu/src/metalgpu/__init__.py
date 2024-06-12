from .interface import Interface, MetalSize
from .operators import sqrt, cos, sin, tan

import platform
assert platform.system() == "Darwin", "[MetalGPU] MetalGPU is only supported on macOS"
