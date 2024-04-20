"""Setup script for install Merlin."""

import os
import sys

# check platform
if (sys.platform != "linux") and (sys.platform != "win32"):
    message = f"Platform {sys.platform} not supported."
    raise OSError(message)

# add current directory to path
sys.path.append(os.path.curdir)

# get setup helpers
from setuptools import setup

# custom setup helpers
from setup_cfg import BuildExt, Extension, copy_dll_libs, ext_options

# create extension
merlin_srclist = [
    "merlin/main.cpp",
    "merlin/wrap_mics.cpp",
    "merlin/wrap_cuda.cpp",
    "merlin/wrap_array.cpp",
    "merlin/wrap_grid.cpp",
    "merlin/wrap_splint.cpp",
    "merlin/wrap_regpl.cpp",
    "merlin/wrap_candy.cpp",
]
merlin_extensions = [Extension("merlin.__init__", merlin_srclist, **ext_options)]

# build extensions and install
if __name__ == "__main__":
    copy_dll_libs()
    setup(
        name="merlin",
        version="1.0.0",
        author="quocdang1998",
        author_email="quocdang1998@gmail.com",
        packages=["merlin"],
        ext_modules=merlin_extensions,
        include_package_data=True,
        python_requires=">=3.6",
        install_requires=["numpy>1.19"],
        extras_require={
            "docs": [
                "Sphinx>5.0",
                "sphinx_rtd_theme>=1.2.0",
                "sphinxcontrib-bibtex",
                "sphinx_design>=0.3.0",
                "breathe>=4.34.0",
                "sphinx-doxysummary>=2.3.2",
            ],
        },
        cmdclass={"build_ext": BuildExt},
    )
