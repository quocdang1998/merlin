Python API
==========

.. raw:: latex

   \setcounter{codelanguage}{0}


Preliminary
-----------

Environment
^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.Environment

Utility
^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.contiguous_to_ndim_idx
   merlin.get_random_subset

Synchronizer
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.Synchronizer


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

Multi-dimensional array
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.array.NdData
   merlin.array.Array
   merlin.array.Parcel
   merlin.array.Stock

Empty array allocator
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.array.empty_array
   merlin.array.empty_parcel
   merlin.array.empty_stock


Interpolation
-------------

Grid
^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.grid.CartesianGrid

Interpolator
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.splint.Method
   merlin.splint.Interpolator


Regression
----------

Polynomial
^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.regpl.Polynomial
   merlin.regpl.new_polynom

Training algorithm
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.regpl.Vandermonde
   merlin.regpl.create_vandermonde

Regressor
^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.regpl.Regressor


Candy API
---------

CP Model
^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.candy.Model
   merlin.candy.load_model
   merlin.candy.Gradient
   merlin.candy.Randomizer

Optimization algorithms
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.candy.Optimizer
   merlin.candy.create_grad_descent
   merlin.candy.create_adagrad
   merlin.candy.create_adam
   merlin.candy.create_adadelta
   merlin.candy.create_rmsprop

Asynchronous fitting
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: pyclass.rst

   merlin.candy.Trainer
