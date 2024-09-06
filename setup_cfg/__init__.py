from .build_ext import BuildExt, Extension
from .copy_lib import copy_dll_libs
from .ext_options import get_extension_options
from .config import MERLIN_VERSION as version

ext_options = get_extension_options()
