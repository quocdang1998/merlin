import os
import sys

# check platform
if (sys.platform != "linux") and (sys.platform != "win32"):
    raise EnvironmentError(f"Platform {sys.platform} not supported.")

# check Cython version
import Cython
from packaging.version import Version, parse
v_cython = parse(Cython.__version__)
v_require = Version("3.0a10")
if v_cython < v_require:
    raise ImportError("Package must be compiled with Cython 3.0")

# import setuptools
from setuptools import Extension, setup
from Cython.Build import cythonize
from setup_cfg import build_ext, ext_options

# extensions
extensions = [
    Extension("merlin.cuda", ["merlin/cuda/core.pyx"],
              language="c++", **ext_options)
]

# build extensions and install
setup(name="merlin",
      version="1.0.0",
      ext_modules=cythonize(extensions, language_level="3str",
                            nthreads=os.cpu_count(), annotate=False),
      cmdclass={"build_ext": build_ext}
)
