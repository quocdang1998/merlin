Python API
==========

.. raw:: latex

   \setcounter{codelanguage}{0}


Environment
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.Environment


Utility
-------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.contiguous_to_ndim_idx
   merlin.get_random_subset

GPU with CUDA
-------------

GPU query
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.cuda.Device
   merlin.cuda.print_gpus_spec
   merlin.cuda.test_all_gpu

CUDA Stream and Event
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.cuda.Stream
   merlin.cuda.Event


Array API
---------

Hidden base class
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.array.NdData

Multi-dimensional array
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.array.Array
   merlin.array.Parcel
   merlin.array.Stock


Grid API
--------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.grid.CartesianGrid
