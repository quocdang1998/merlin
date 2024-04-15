import os
import sys
import glob

import numpy as np

from .config import *

def get_extension_options():
    # options for building Cython extension
    module_dir = os.path.abspath(os.path.join(__file__, "../.."))
    ext_options = dict()

    # language
    ext_options["language"] = "c++"

    # include directory
    ext_options["include_dirs"] = [os.path.join(module_dir, "src")]
    ext_options["include_dirs"] += [np.get_include()]

    # compile macros
    numpy_macro = ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
    ext_options["define_macros"] = [numpy_macro]
    if MERLIN_CUDA:
        ext_options["define_macros"] += [("__MERLIN_CUDA__", None)]
    if MERLIN_LIBKIND == "STATIC":
        ext_options["define_macros"] += [("__MERLIN_BUILT_AS_STATIC__", None)]

    # extra compile arguments
    if sys.platform == "linux":
        ext_options["extra_compile_args"] = ["-std=c++20",
                                             "-Wno-unused-but-set-variable"]
    elif sys.platform == "win32":
        ext_options["extra_compile_args"] = ["-std:c++20",
                                             "/wd4251", "/wd4551"]

    # dependancies
    depends = glob.glob(os.path.join(module_dir, "setup_cfg", "*.py"))
    if sys.platform == "linux":
        depends += glob.glob(os.path.join(module_dir, "build", "libmerlin*"))
    elif sys.platform == "win32":
        depends += [os.path.join(module_dir, "build", "merlin.lib")]
        if MERLIN_LIBKIND == "SHARED":
            depends += [os.path.join(module_dir, "build", "merlin.dll")]
    ext_options["depends"] = depends

    # extra link options
    if sys.platform == "win32":
        ext_options["extra_link_args"] = ["/NODEFAULTLIB:LIBCMT.lib",
                                          "/IGNORE:4286"]

    # link librairies
    ext_options["libraries"] = ["merlin"]
    if MERLIN_CUDA:
        ext_options["libraries"] += ["merlincuda"]
    ext_options["libraries"] += ["merlinrdc", "merlinenv"]
    if MERLIN_CUDA:
        ext_options["libraries"] += ["cudart_static", "cudadevrt", "cuda"]
    if MERLIN_DEBUG and (sys.platform == "win32"):
        ext_options["libraries"] += ["DbgHelp"]

    # library directory
    ext_options["library_dirs"] = [os.path.join(module_dir, "build")]
    if MERLIN_CUDA:
        ext_options["library_dirs"] += [CUDALIB]

    # runtime library
    if (sys.platform == "linux"):
        rt_dir = "${ORIGIN}"
        ext_options["runtime_library_dirs"] = [rt_dir]
        if MERLIN_CUDA:
            ext_options["runtime_library_dirs"] += [CUDALIB]

    return ext_options
