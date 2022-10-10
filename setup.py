import os

from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("merlin", ["merlin/main.pyx"],
              include_dirs=["src"],
              library_dirs=["build"], libraries=["libmerlin.so", "merlincuda"],
              runtime_library_dirs=["build"],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              language="c++"
              )
]

setup(name="merlin",
      version="1.0.0",
      ext_modules=cythonize(extensions,
                            language_level="3str",
                            nthreads=os.cpu_count(),
                            annotate=False))

