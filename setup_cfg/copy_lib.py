"""Copy dynamic linked library from the C++ interface to the Python source
directory."""

import os
import sys
from shutil import copyfile

from .config import *


def copy_dll_libs():
    py_src_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../..")), "merlin")
    if sys.platform == "win32":
        merlinshared_lib = "merlinenv.dll"
        merlin_lib = "merlin.dll"
    elif sys.platform == "linux":
        merlinshared_lib = "libmerlinenv.so"
        merlin_lib = "libmerlin.so"
    merlinshared_src = os.path.join(MERLIN_BIN_DIR, merlinshared_lib)
    merlinshared_dst = os.path.join(py_src_dir, merlinshared_lib)
    copyfile(merlinshared_src, merlinshared_dst)
    if MERLIN_LIBKIND == "SHARED":
        merlin_src = os.path.join(MERLIN_BIN_DIR, merlin_lib)
        merlin_dst = os.path.join(py_src_dir, merlin_lib)
        copyfile(merlin_src, merlin_dst)
