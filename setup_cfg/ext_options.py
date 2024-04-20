import os
import sys
import glob

import numpy as np

from .config import *


def get_extension_options():
    # options for building extension
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
        ext_options["extra_compile_args"] = ["-std=c++20", "-Wno-unused-but-set-variable"]
        ext_options["extra_compile_args"] += ["-fopenmp", "-flto=auto", "-fno-fat-lto-objects"]
    elif sys.platform == "win32":
        ext_options["extra_compile_args"] = ["/std:c++20", "/wd4251", "/wd4551"]
        ext_options["extra_compile_args"] += ["-openmp:llvm"]

    # dependencies
    depends = glob.glob(os.path.join(module_dir, "setup_cfg", "*.py"))
    if sys.platform == "linux":
        depends += glob.glob(os.path.join(module_dir, "build", "libmerlin*"))
    elif sys.platform == "win32":
        depends += [os.path.join(module_dir, "build", "merlin.lib")]
        if MERLIN_LIBKIND == "SHARED":
            depends += [os.path.join(module_dir, "build", "merlin.dll")]
    ext_options["depends"] = depends

    # extra link options
    if sys.platform == "linux":
        ext_options["extra_link_args"] = ["-flto=auto", "-fno-fat-lto-objects"]
    elif sys.platform == "win32":
        ext_options["extra_link_args"] = ["/NODEFAULTLIB:LIBCMT.lib", "/IGNORE:4286"]
        if MERLIN_DEBUG:
            ext_options["extra_link_args"] += ["/NODEFAULTLIB:MSVCRT.lib"]

    # link libraries
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
        ext_options["library_dirs"] += CUDADIR

    # runtime library
    if sys.platform == "linux":
        rt_dir = "${ORIGIN}"
        ext_options["runtime_library_dirs"] = [rt_dir]
        if MERLIN_CUDA:
            ext_options["runtime_library_dirs"] += CUDADIR

    # CUDA device linker arguments
    if MERLIN_CUDA:
        ext_options["cuda"] = True
        ext_options["nvcc_executable"] = NVCC
        ext_options["cuda_arch"] = CUDA_ARCHITECHTURE
        ext_options["cuda_linkdir"] = CUDADIR + [MERLIN_BIN_DIR]
        ext_options["lib_cudart"] = CUDART
        ext_options["lib_cudadevrt"] = CUDADEVRT
        ext_options["lib_cudadriver"] = CUDADRIVER
        ext_options["libs_device_linker"] = CUDA_STANDARD_LIBRARIES.strip().split()
        if sys.platform == "linux":
            ext_options["libs_device_linker"] += ["libmerlinrdc.a", "libmerlincuda.a"]
        elif sys.platform == "win32":
            ext_options["libs_device_linker"] += ["merlinrdc.lib", "merlincuda.lib", "merlin.lib"]
    else:
        ext_options["cuda"] = False

    return ext_options
