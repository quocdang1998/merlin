import os
import sys

# check platform
if (sys.platform != "linux") and (sys.platform != "win32"):
    raise EnvironmentError(f"Platform {sys.platform} not supported.")

# add current folder to search path
package_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(package_dir)
abspath_wrt_package = lambda p : os.path.join(package_dir, p)

# check Cython version
import Cython
from packaging.version import Version, parse
v_cython = parse(Cython.__version__)
v_require = Version("3.0.0a10")
if v_cython < v_require:
    raise ImportError("Package must be compiled with Cython version later than 3.0.0a10")

# import setuptools
from setuptools import Extension, setup
from Cython.Build import cythonize
from setup_cfg import build_ext, ext_options, copy_dll_libs

# Cython extensions
extensions = [
    Extension("merlin.env", ["merlin/env.pyx"],
              language="c++", **ext_options),
    Extension("merlin.cuda", ["merlin/cuda/core.pyx"],
              language="c++", **ext_options),
    Extension("merlin.array", ["merlin/array/core.pyx"],
              language="c++", **ext_options),
    Extension("merlin.splint", ["merlin/splint/core.pyx"],
              language="c++", **ext_options)
]

# build extensions and install
if __name__ == "__main__":
    copy_dll_libs()
    cython_extensions = cythonize(extensions, language_level="3str",
                                  include_path=[package_dir],
                                  nthreads=os.cpu_count(), annotate=False)
    for ext in cython_extensions:
        ext.depends = []

    setup(name="merlin",
          version="1.0.0",
          author="quocdang1998",
          author_email="quocdang1998@gmail.com",
          packages=["merlin"],
          ext_modules=cython_extensions,
          include_package_data=True,
          python_requires=">=3.6",
          install_requires=["numpy>1.19"],
          extras_require={
              "docs": ["Sphinx>5.0", "sphinx_rtd_theme>=1.2.0",
                       "sphinxcontrib-bibtex", "sphinx_design>=0.3.0",
                       "breathe>=4.34.0", "sphinx-doxysummary>=2.3.2"]
          },
          cmdclass={"build_ext": build_ext})
