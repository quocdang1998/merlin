import os
import sys
import glob

# check platform
if (sys.platform != "linux") and (sys.platform != "win32"):
    raise EnvironmentError(f"Platform {sys.platform} not supported.")

# import setuptools
from setuptools import Extension, setup
from setup_cfg import build_ext, ext_options, copy_dll_libs

#import pybind11
from pybind11.setup_helpers import Pybind11Extension
merlin_extensions = [
    Pybind11Extension("merlin", glob.glob(os.path.join("merlin", "*.cpp")), **ext_options)
]


# build extensions and install
if __name__ == "__main__":
    copy_dll_libs()

    setup(name="merlin",
          version="1.0.0",
          author="quocdang1998",
          author_email="quocdang1998@gmail.com",
          packages=["merlin"],
          ext_modules=merlin_extensions,
          include_package_data=True,
          python_requires=">=3.6",
          install_requires=["numpy>1.19"],
          extras_require={
              "docs": ["Sphinx>5.0", "sphinx_rtd_theme>=1.2.0",
                       "sphinxcontrib-bibtex", "sphinx_design>=0.3.0",
                       "breathe>=4.34.0", "sphinx-doxysummary>=2.3.2"]
          },
          cmdclass={"build_ext": build_ext}
    )
