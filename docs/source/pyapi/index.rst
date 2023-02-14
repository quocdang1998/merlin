Python API
==========

.. raw:: latex

   \setcounter{codelanguage}{0}


GPU with CUDA
-------------

GPU query
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.cuda.Device
   merlin.cuda.DeviceLimit
   merlin.cuda.print_all_gpu_specification
   merlin.cuda.test_all_gpu

Manage CUDA Context
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.cuda.Context
   merlin.cuda.ContextFlags
   merlin.cuda.create_primary_context

CUDA Stream and Event
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.cuda.Stream
   merlin.cuda.StreamSetting
   merlin.cuda.Event
   merlin.cuda.EventCategory
   merlin.cuda.record_event

Multi-dimensional array
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.array.Array
