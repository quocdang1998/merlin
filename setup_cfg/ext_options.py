import os
import sys
import glob

from .config import *

def get_extension_options():
    # options for building Cython extension
    module_dir = os.path.abspath(os.path.join(__file__, "../.."))
    ext_options = dict()

    # include directory
    ext_options["include_dirs"] = [os.path.join(module_dir, "src")]

    # compile macros
    numpy_macro = ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
    ext_options["define_macros"] = [numpy_macro]
    if MERLIN_CUDA:
        ext_options["define_macros"] += [("__MERLIN_CUDA__", None)]
    if MERLIN_LIBKIND == "STATIC":
        ext_options["define_macros"] += [("__MERLIN_BUILT_AS_STATIC__", None)]

    # extra compile arguments
    if (sys.platform == "linux"):
        ext_options["extra_compile_args"] = ["-std=c++17",
                                             "-Wno-unused-but-set-variable"]
    elif (sys.platform == "win32"):
        ext_options["extra_compile_args"] = ["-std:c++17"]
        if MERLIN_LIBKIND == "SHARED":
            ext_options["extra_compile_args"] += ["/wd4251", "/wd4551"]

    # dependancies
    depends = glob.glob(os.path.join(module_dir, "setup_cfg", "*.py"))
    if (sys.platform == "linux"):
        depends += glob.glob(os.path.join(module_dir, "build", "libmerlin.*"))
    elif (sys.platform == "win32"):
        depends += [os.path.join(module_dir, "build", "merlin.lib")]
        if MERLIN_LIBKIND == "SHARED":
            depends += [os.path.join(module_dir, "build", "merlin.dll")]
    ext_options["depends"] = depends

    # extra link options
    if (sys.platform == "win32"):
        ext_options["extra_link_args"] = ["/NODEFAULTLIB:LIBCMT.lib",
                                          "/IGNORE:4286"]

    # link librairies
    ext_options["libraries"] = ["merlin"]
    if MERLIN_LIBKIND == "SHARED":
        ext_options["libraries"] += ["merlincuda"]
    if MERLIN_CUDA:
        ext_options["libraries"] += ["cudart_static", "cudadevrt", "cuda"]

    # library directory
    ext_options["library_dirs"] = [os.path.join(module_dir, "build")]
    if MERLIN_CUDA:
        ext_options["library_dirs"] += [CUDALIB]

    # runtime library
    if (sys.platform == "linux") and (MERLIN_LIBKIND == "SHARED"):
        rt_dir = os.path.join(module_dir, "build")
        ext_options["runtime_library_dirs"] = [rt_dir]
        if MERLIN_CUDA:
            ext_options["runtime_library_dirs"] += [CUDALIB]

    return ext_options
