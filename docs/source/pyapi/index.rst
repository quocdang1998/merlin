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

   merlin.env.Environment


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
   merlin.cuda.print_gpus_spec
   merlin.cuda.test_all_gpu

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

Interpolator API
----------------

Grid
^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.splint.CartesianGrid

Polynomial interpolation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.splint.Interpolator
   merlin.splint.Method

