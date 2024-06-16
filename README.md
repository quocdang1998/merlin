# Merlin - Compression And Interpolation Of Nuclear Homogenized Cross Section Data

The Merlin project aims to provide a simple data processing framework for few-group homogenized cross sections dataset, which is utilized in industrial reactor simulation and reactor research projects. It is an **independent**, **portable**, and **scalable** library written in modern C++20 and CUDA, with a Python interface installable with ``pip``.

Merlin consists of two main libraries: ``candy`` for compression and decompression of the cross section dataset with tensor decomposition by the gradient-based approach of CANDECOMP-PARAFAC, and ``splint`` for interpolation of the reconstructed tensor by linear or polynomial interpolation.

Merlin is optimized with many optimization techniques, including vectorization with AVX and parallelization on both CPU and multi-GPUs systems, to expedite its algorithms.

## Installation

### C++/CUDA library

Prior to installing, please ensure these prerequisites:
* ``gcc>=11.2`` (on Linux)
* MSVC 2022 ``cl.exe>=19.39`` (on Windows)
* CUDA ``nvcc>=12.1`` (if compiling library for GPU)
* CMake not older than ``3.25``
* Build system: ``GNU make`` for Linux, or ``Ninja`` for Windows

Configuration and compilation:
```
cmake --preset=linux .  # switch to --preset=windows to compile with MSVC
cd build
# ninja  # for Windows
make -j 8
```

Installation:

```
cmake --install . --prefix="/path/to/install folder"
```

> Due the restriction of the CUDA compiler ``nvcc``, Merlin can only be compiled using ``gcc`` on Linux or MSVC 2022 on Windows. Other compilers such as ``clang`` on Mac OS, or Intel compiler ``icc``, will raise compilation errors with Merlin.

### Python interface

Python interface requires the compiled **C++/CUDA library**. It enables calling C++/CUDA functions and classes from Python scripts through ``Pybind11``.

```
pip install .
```

To compile the Python interface "inplace" (binary module copied to the source directory):

```
python setup.py build_ext -i
```

## Documentation

Required Sphinx extensions:

```
pip install -U sphinx_rtd_theme sphinx_design breathe sphinx_doxysummary
```

Code documentation can be built with Doxygen and Sphinx:

```
cd docs
doxygen Doxyfile
make html  # or .\make.bat html on Windows
```

## License

This project is subject to the MIT License.
