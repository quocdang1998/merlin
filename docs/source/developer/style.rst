Code styling
============

The Google's style guide is applied to C++/CUDA sources. The style check is
performed by ``cpplint``.

.. code-block:: sh

   cpplint --recursive --extensions=hpp,cpp,tpp,cu,rdc src

Besides the standard Google style rules, some additional rules are also applied:

*  Source files always starts with the copyright comment. Developer must enter
   the year followed by her/his name (see the example below).

*  Each include must be followed by classe/function names called from included
   header, listed in a line comment. There are 4 categories of include files,
   separated by blank lines. Order of include is:

   #. C header files, like ``cstdio``, ``cstdlib``, ``cstring``, ``cstdint``.

   #. C++ header files, like ``iostream``, ``string``, ``tuple``, ``vector``.

   #. External libraries' header files, like ``gsl/gsl_foo.h``, ``fmt/core.h``.

   #. Project's header files, like ``merlin/utils.hpp``, ``merlin/logger.hpp``.

   .. code-block:: c++
      :linenos:

      // Copyright 2022 quocdang1998

      #include <cstring>  // std::memcpy

      #include <initializer_list>  // std::inializer_list
      #include <vector>  // std::vector

      #include "cuda.h"  // cuDeviceGetAttribute

      #include "merlin/logger.hpp"  // MESSAGE, FAILURE
      #include "merlin/array/array.hpp"  // merlin::array::Array

*  Declarations and definitions must be enclosed in namespace ``merlin``. The
   use of ``using namspace`` is prohibited except in executable source file
   (inside the ``main`` function).

   .. code-block:: c++
      :linenos:

      namespace merlin {

      // your code goes here

      }  // namespace merlin

*  Indent is 4 spaces by default (contrary to the default 2 spaces applied by
   Google's rules). Elements in between header guards, namespaces and macros
   keep the same indent level as its parents. Class encapsulation indicators
   (``public``, ``protected`` and ``private``) are indented by 2 spaces.

   .. code-block:: c++
      :linenos:

       // foo.hpp
      #ifndef FOO_HPP_
      #define FOO_HPP_

      namespace merlin {

      #ifdef __NVCC__
      void foo(void);
      #endif  // __NVCC__

      class Foo {
        public:
          void public_method(void);

        protected:
          void protected_method(void);

        private:
          void private_method(void);
      };

      }  // namespace merlin

      #endif  // FOO_HPP_

