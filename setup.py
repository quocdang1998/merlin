import os
import sys
import glob

from setuptools import Extension, setup
from Cython.Build import cythonize
from packaging.version import Version, parse

# check Cython version
import Cython
v_cython = parse(Cython.__version__)
v_require = Version("3.0a1")
if (v_cython < v_require):
    raise ImportError("Package must be compiled with Cython 3.0")

# import variable from config.py (generated file by CMake)
from config import *

# construct Cython extension options
module_dir = os.path.dirname(os.path.realpath(__file__))
ext_options = dict()
ext_options["include_dirs"] = [os.path.join(module_dir, "src")]
ext_options["library_dirs"] = [os.path.join(module_dir, "build")]
if (MERLIN_CUDA == "ON") and (MERLIN_LIBKIND == "SHARED"):
    ext_options["libraries"] = ["merlin", "merlincuda"]
else:
    ext_options["libraries"] = ["merlin"]

# add system-dependant options
if (sys.platform == "linux"):  # on Linux
    # std norm
    ext_options["extra_compile_args"] = ["-std=c++17"]
    # add runtime library
    ext_options["runtime_library_dirs"] = [os.path.join(module_dir, "build")]
    # for recompile if libraries are updated
    ext_options["depends"] = glob.glob(os.path.join(module_dir, "build", "libmerlin*"))
elif (sys.platform == "win"):  # on Windows
    # std norm
    ext_options["extra_compile_args"] = ["/std=c++17"]
else:
    printf(f"Unsupported OS {sys.platform}.")

extensions = [
    Extension("merlin.device", ["merlin/device/core.pyx"],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              language="c++", **ext_options)
]

setup(name="merlin",
      version="1.0.0",
      ext_modules=cythonize(extensions,
                            language_level="3str",
                            nthreads=os.cpu_count(),
                            annotate=False))
