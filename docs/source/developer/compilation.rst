Compilation scheme
==================

Environment library
-------------------

In the circumstances of the Python interface, C++ extern variables and class static attributes need to be shared between
all compiled binary extensions. Different Python binary modules (which are dynamic linked libraries) must refer to the
same object when they call the same symbolic name. Thus, these variables and attributes must be isolated inside a shared
library.

.. image:: ../_img/env_comp.*
   :align: center
   :alt: Environment library compilation

The library ``libmerlinenv`` is always compiled regardless of the kind of the chosen library or the CUDA option.

Relocatable device code
-----------------------

In CUDA separate compilation, device functions (functions run exclusively on GPUs) must always be linked in static
library, so the executable contains all device functions when invoked. The reason is because GPU is unable to access
libraries in the storage at run-time. For that reason, CUDA device and host-device functions must be compiled separately
and stored inside a static library. In Merlin, these functions muts be defined in ``.rdc`` files, which stand for
"Relocatable Device Code". The compilation scheme is as following:

   .. image:: ../_img/rdc_comp.*
      :align: center
      :alt: Relocatable device code compilation

Since ``libmerlinrdc`` contains both CPU and GPUs functions, it is always compiled regardless of the kind of the chosen
library or the CUDA option. In non-CUDA mode, host-device functions degenerate to pure CPU functions, rendering the
library no different from a regular static library.

In order to avoid the function calling overhead caused by separate unit compilation, all source files in this library is
compiled with **link-time optimization**. When switched on, instead of translating directly the C++/CUDA source files
into binary, the compiler will compile them into **Intermediate Representation** (IR). These IR are native transcription
of the source to the compiler, and the real optimization can be performed at link time when these IR are compiled into
binaries.

CUDA global library
-------------------

CUDA introduces a category of functions annotated with the __global__ decorator. These are functions are specifically
designed to run on GPU, but remain callable from CPU. Within the Merlin CUDA interface, they serve as crucial
components, fulfilling roles like resource allocator, device function invocator, and memory optimizer. All interactions
between Merlin GPU functions and the GPU occur through these functions.

.. image:: ../_img/cuda_comp.*
   :align: center
   :alt: CUDA global function compilation

Similar to CUDA ``__device__`` and ``__host__ __device__`` functions, global functions require static linking. However,
they must be organized in a separate library of the relocatable code library because of the **device linker resolution**
step (see `Separable Compilation <https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html>`_).
During this step, the CUDA linker resolves all device functions invoked by global functions and effectively replaces all
relocatable code by either inlining them or by transforming them into absolute address code. Without the library
separation, other executables and libraries cannot link to the relocatable code, and thus have to recompile the whole
relocatable code library. For that reason, CUDA global functions undergo compilation within a separate static library
with an added device linker step.

Since ``libmerlincuda`` only encompasses CUDA functions, it is compiled exclusively when CUDA option is enabled. Apart
from CUDA global functions, this library also contains C++ wrapper functions calling to those global functions.

Main library compilation
------------------------

The main library of Merlin encompasses all others ``.cpp`` and ``.cu`` (in CUDA mode) sources. This library acts as the
intermediary between the C++ interface and the lower-level parallel C++/CUDA library. Users have the flexibility to
compile the main library as either a static or dynamic library, with both options supported on Linux and Windows
platforms.

.. image:: ../_img/main_comp.*
   :align: center
   :alt: Main library compilation

Although the compilation with Cmake supports both the compilation of static library and dynamic library, it is
recommended to use dynamic library on Linux and static library on Windows for speed and simplicity.
