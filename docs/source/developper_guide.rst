Developper Guide
================

C++ application
---------------

The code style of C++ code is Google's style guide, checked by ``cpplint``.

.. code-block:: sh

   cpplint --root=inc inc/merlin/*.hpp
   cpplint src/*.hpp

General rule for source files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  Header files always starts with the copyright comment.

   .. code-block:: c++
      :linenos:

      // Copyright 2022 quocdang1998

*  Each include must be followed by classes/functions called from included
   library, listed in a line comment. There are 4 group of include files, each
   group must be put between 2 blank lines. Order of include group is:

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

      #include "merlin/logger.hpp"  // MESSAGE, FAILURE

*  There are at least 2 spaces between a line comment and the code.

*  All declaration and definition must be enclosed in namespace. The use of
   ``using namspace`` is highly unrecommended.

   .. code-block:: c++
      :linenos:

      namespace merlin {

      // your code here

      }  // namespace merlin

*  Yes
