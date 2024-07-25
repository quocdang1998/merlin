User Guide
==========

Multi-dimensional array
-----------------------

Merlin utilizes its proprietary multi-dimensional array implementation, offering a straightforward user interface for
transferring data between computer storage and RAM, or between RAM and GPU global memory. Each elements of the array
must be in **double precision**, otherwise, the library will raise an error.

.. note::

   In order to speed up the calculation process, number of dimension is limited to 16. User can change this upper limit
   through the variable :cpp:var:`merlin::max_dim` and recompile the whole library.

CPU memory
^^^^^^^^^^

Classes :cpp:class:`merlin::array::Array` (in C++) and :py:class:`merlin.array.Array` (in Python) represents a
multi-dimensional array allocated on CPU. It also allow user to associate the pointer address directly to a
pre-allocated memory, thus avoiding unnecessary copies of the data.

.. tab-set-code::

   .. code-block:: c++

      #include "merlin/array/array.hpp"  // merlin::array::Array
      #include "merlin/vector.hpp"       // merlin::UIntVec

      // example of a flatten data in memory
      double data[6] = {
          0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
          12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
      };
      merlin::UIntVec shape = {2, 3, 4};  // data shape

      // data strides
      merlin::UIntVec c_strides = {  // C-contiguous
          shape[1] * shape[2] * sizeof(double),
          shape[2] * sizeof(double),
          sizeof(double)
      };
      merlin::UIntVec f_strides = {  // Fortran-contiguous
          sizeof(double),
          shape[0] * sizeof(double),
          shape[1] * shape[0] * sizeof(double)
      };
      // direct assignment (without data copy)
      merlin::array::Array a(data, shape, c_strides, false);

   .. code-block:: python

      import numpy as np
      from merlin.array import Array

      # convert NumPy array to Merlin Array
      np_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                         dtype=np.float64)   # dtype MUST be np.float64
      a1 = Array(np_data, copy=False)  # direct assignment
      a2 = Array(np_data, copy=True)   # copy to another place

      # convert Merlin Array back to NumPy
      np_a1 = np.array(a1, copy=False)
      np_a2 = mp.array(a2, copy=True)

.. note::

   Please note that the class ``Array`` only supports positive strided array. Negative strided array (such as
   ``np_data[::-1,:]``) will result in an undefined-behavior error or even a segmentation fault.

GPU memory
^^^^^^^^^^

Multi-dimensional data on GPU is handled by classes :cpp:class:`merlin::array::Parcel` (in C++) and
:py:class:`merlin.array.Parcel` (in Python). To transfer data to GPU, user have to convert their array memory into
Merlin's ``Array`` first, then allocate and transfer data to/from GPU.

.. tab-set-code::

   .. code-block:: c++

      #include "merlin/array/array.hpp"   // merlin::array::Array
      #include "merlin/array/parcel.hpp"  // merlin::array::Parcel
      #include "merlin/cuda/device.hpp"   // merlin::cuda::Device
      #include "merlin/cuda/stream.hpp"   // merlin::cuda::Stream

      // for GPU specifications, please refer to merlin::cuda::print_gpus_spec
      merlin::cuda::Device gpu(0);  // get GPU
      gpu.set_as_current();         // set as current GPU
      // initialize CUDA asynchronous stream
      merlin::cuda::Stream synch_stream(merlin::cuda::StreamSetting::NonBlocking);

      merlin::array::Array a = ...;                      // construct an Array
      merlin::array::Parcel p(a.shape(), synch_stream);  // allocate data
      p.transfer_data_to_gpu(a, synch_stream);           // copy data to GPU
      ...                                                // do something
      a.clone_data_from_gpu(p, synch_stream);            // copy data from GPU

   .. code-block:: python

      from merlin.array import Array, empty_parcel
      from merlin.cuda import Device, Stream

      # for GPU specifications, please refer to merlin.cuda.print_gpus_spec
      gpu = Device(0)      # get GPU
      gpu.set_as_current() # set as current GPU
      # initialize CUDA asynchronous stream
      synch_stream = Stream(setting="nonblocking")

      a = Array...               # construct an Array
      p = empty_parcel(a.shape, synch_stream)  # allocate data
      p.transfer_data_to_gpu(a, synch_stream)  # copy data to GPU
      ...                                      # do something
      a.clone_data_from_gpu(p, synch_stream)   # copy data from GPU

.. note::

   If Merlin is not compiled using CUDA, an error will be thrown when trying to use ``Parcel``'s methods.

Out-of-core memory
^^^^^^^^^^^^^^^^^^

Multi-dimensional out-of-core data (i.e. data resides on in the storage of the computer) can be interacted with using
classes :cpp:class:`merlin::array::Stock` (in C++) and :py:class:`merlin.array.Stock` (in Python).

.. tab-set-code::

   .. code-block:: c++

      #include "merlin/array/array.hpp"   // merlin::array::Array
      #include "merlin/array/stock.hpp"   // merlin::array::Stock

      // read data from storage to RAM
      merlin::array::Stock s("/path/to/filename");  // // assign to a file
      merlin::array::Array a(s.shape());            // allocate data on CPU
      a.extract_data_from_file(s);                  // read data

      // save data from RAM to storage
      merlin::array::Array a = ...
      merlin::array::Stock s("/path/to/filename", a.shape());  // allocate a file
      s.record_data_to_file(a);                                // save data

   .. code-block:: python

      from merlin.array import Array, Stock, empty_array, empty_stock

      # read data from storage to RAM
      s = Stock("/path/to/filename")  # assign to a file
      a = empty_array(s.shape)        # allocate data on CPU
      a.extract_data_from_file(s)     # read data

      # save data from RAM to storage
      a = ...
      s = empty_stock(a.shape)   # allocate a file
      s.record_data_to_file(a);  # save data

.. note::

   When the stock file is copied between system with different endian (e.g. copy from a little-endian system to a
   big-endian system), each element of the subsequent ``Array`` must be bit-flipped.


Data compression
----------------

Merlin achieves data compression by using CANDECOMP-PARAFAC (CP) model :cite:p:`acar2011scalable`. This model assumes
that the few-group homogenized cross section can be decomposed into:

.. math::

   \hat{\boldsymbol{\sigma}} \approx \sum_{r=0}^{R-1} \left( \boldsymbol{v}_{r,0} \otimes \boldsymbol{v}_{r,1} \otimes
   \dots \otimes \boldsymbol{v}_{r,D-1} \right)


Notation explanation:

================================= ===========================================================
Variable                          Description
================================= ===========================================================
:math:`\hat{\boldsymbol{\sigma}}` Few-group homogenized cross section.
:math:`R`                         Decomposition rank of the model (hyper-parameter).
:math:`D`                         Number of dimension of the data.
:math:`\boldsymbol{v}_{r,d}`      Eigenvector of the model at rank :math:`r`, axis :math:`d`.
:math:`\otimes`                   Cartesian product between 2 tensors.
================================= ===========================================================

To find optimal values of elements of eigenvectors :math:`\boldsymbol{v}_{r,d}`, the relative mean square error is used:

.. math::

   L_r = \frac{1}{2} \sum_{\boldsymbol{i}} \left[\frac{1}{\hat{\boldsymbol{\sigma}}[\boldsymbol{i}]}
   \left( \hat{\boldsymbol{\sigma}}[\boldsymbol{i}] - \sum_{r=0}^{R-1} \left(\prod_{d=0}^{D-1}
   \boldsymbol{v}_{r,d}[i_d]\right) \right) \right]^2

in which :math:`\boldsymbol{i} = (i_0, i_1, \dots, i_d, \dots, i_{D-1})^\intercal` represents a multi-dimensional index
vector, :math:`\hat{\boldsymbol{\sigma}}[\boldsymbol{i}]` represents the few-group homogenized cross section at the
index :math:`\boldsymbol{i}`, and :math:`\boldsymbol{v}_{r,d}[i_d]` represents the :math:`i_d`-th element of the
eigenvector :math:`\boldsymbol{v}_{r,d}`.

Another loss function also supported by Merlin is the absolute mean square error:

.. math::

   L_a = \frac{1}{2} \sum_{\boldsymbol{i}} \left[ \hat{\boldsymbol{\sigma}}[\boldsymbol{i}] - \sum_{r=0}^{R-1}
   \left(\prod_{d=0}^{D-1} \boldsymbol{v}_{r,d}[i_d]\right) \right]^2

From this loss function, one can calculate the gradient wrt. each entry of the eigenvectors
:math:`\boldsymbol{v}_{r,d}[i_d]`. For example, the gradient of the relative mean square error can be formulated as:

.. math::

   \frac{\partial L_r}{\partial \boldsymbol{v}_{r,d}[i_d]} = \sum_{\substack{\boldsymbol{i}' \\ i'_d = i_d}} \left[
   \frac{1}{{\hat{\boldsymbol{\sigma}}[\boldsymbol{i}']}^2} \left(\prod_{\substack{d'=0 \\ d' \ne d}}^{D-1}
   \boldsymbol{v}_{r,d'}[{i'}_{d'}]\right) \left(\sum_{r'=0}^{R-1} \left(\prod_{d'=0}^{D-1}
   \boldsymbol{v}_{r',d'}[{i'}_{d'}]\right) - \hat{\boldsymbol{\sigma}}[\boldsymbol{i'}]\right) \right]

Since the gradient always points in the direction of greatest increase, the CP model can be updated in the inverse
direction to minimize the loss function. Many gradient descents algorithms are proposed. Merlin supports five
algorithms: stochastic gradient descent :cite:p:`amari1993backpropagation`, adaptive gradient
:cite:p:`duchi2011adaptive`, adaptive moment gradient :cite:p:`zhang2018improved`, adaptative delta
:cite:p:`zeiler2012adadelta` and root mean square propagation :cite:p:`liu2022hyper`.

Within Merlin, the CP model undergoes an update using the back-propagation algorithm for :math:`K` iterations, then the
error :math:`L_r` before and after training is compared against each other. If their relative difference falls below a
threshold :math:`\alpha`, the gradient descent process halts. Otherwise, it iterates for another :math:`K` iterations
and repeats the comparison until the stop criterion is met. All actions listed above are encompasses in classes
:cpp:class:`merlin::candy::Trainer` (in C++) and :py:class:`merlin.candy.Trainer` (in Python).

.. tab-set-code::

   .. code-block:: c++

      #include "merlin/array/array.hpp"      // merlin::array::Array
      #include "merlin/candy/model.hpp"      // merlin::candy::Model
      #include "merlin/candy/optimizer.hpp"  // merlin::candy::Optimizer, merlin::candy::create_grad_descent
      #include "merlin/synchronizer.hpp"     // merlin::Synchronizer, merlin::ProcessorType
      #include "merlin/candy/trainer.hpp"    // merlin::candy::Trainer

      merlin::array::Array data = ...;   // initialize data
      merlin::candy::Model model = ...;  // initialize model

      // create an optimizer
      merlin::candy::Optimizer optimizer = merlin::candy::create_grad_descent(...);
      // create an asynchronous stream to work
      merlin::Synchronizer synch_stream(merlin::ProcessorType::Cpu);
      // initialize a trainer to train CP model
      merlin::candy::Trainer train(model, optimizer, synch_stream);
      // train model with K = 10000, alpha = 1e-2 using 4 threads
      train.update(data, 10000, 1e-2, 4);
      synch_stream.synchronize();  // stop the main thread until the training algorithm has finished

   .. code-block:: python

      from merlin import Synchronizer
      from merlin.array import Array
      from merlin.candy import Model, create_grad_descent, Trainer

      data = Array(...)   # initialize data
      model = Model(...)  # initialize model

      # create an optimizer
      optimizer = create_grad_descent(...)
      #  create an asynchronous stream to work
      synch_stream = Synchronizer("cpu")
      # initialize a trainer to train CP model
      train = Trainer(model, optimizer, synch_stream)
      # train model with K = 10000, alpha = 1e-2 using 4 threads
      train.update_cpu(data, 10000, 1e-2, 4);
      synch_stream.synchronize();  # stop the main thread until the training algorithm has finished

.. note::

   The default behavior of the ``Trainer`` class is asynchronous, thus allowing simultaneous training of multiple CP
   models on multiple datasets. Python users must ensure synchronization of ``Trainer`` objects before they are no
   longer referred or contained by any variables, and before the script ends. Destroying un-synchronized ``Trainer``'s
   results in segmentation fault.

Polynomial representation
-------------------------

Interpolation
^^^^^^^^^^^^^

Merlin provides linear and polynomial interpolation library under the sub-library ``splint``. It facilitates the
construction of coefficients and evaluation of polynomial interpolation on Cartesian grids in parallel using CPU and
GPUs.

.. tab-set-code::

   .. code-block:: c++

      #include <vector>

      #include "merlin/array/array.hpp"          // merlin::array::Array
      #include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
      #include "merlin/splint/interpolator.hpp"  // merlin::splint::Interpolator, merlin::splint::Method
      #include "merlin/synchronizer.hpp"         // merlin::Synchronizer, merlin::ProcessorType
      #include "merlin/vector.hpp"               // merlin::DoubleVec

      merlin::array::Array data = ...;         // initialize data
      merlin::grid::CartesianGrid grid = ...;  // initialize grid
      std::Vector<merlin::splint::Method> methods = {  // interpolation method on each dimension
          merlin::splint::Method::Newton,
          merlin::splint::Method::Linear,
          merlin::splint::Method::Newton,
          merlin::splint::Method::Lagrange,
          ...
      }

      // create interpolator
      merlin::Synchronizer synch_stream(merlin::ProcessorType::Cpu);
      merlin::splint::Interpolator interp(grid, data, methods.data(), synch_stream);
      interp.build_coefficients(4);  // build coefficients with 4 threads in the background

      // interpolation
      merlin::array::Array points = ...      // points has shape [npoints, ndim] and must be C-contiguous
      // merlin::array::Parcel points = ...  // use GPU array if the synch_stream is on GPU
      merlin::DoubleVec result(npoints);     // initialize memory for storing interpolation result
      interp.evaluate(points, result, 8);    // asynchronous interpolation using 8 threads
      synch_stream.synchronize();            // for the main thread to wait until all tasks finished

   .. code-block:: python

      from merlin import Synchronizer
      from merlin.array import Array
      from merlin.grid import CartesianGrid
      from merlin.splint import Interpolator, Method

      data = Array(...)          # initialize data
      grid = CartesianGrid(...)  # initialize grid
      methods = [                # interpolation method on each dimension
          Method.Newton,
          Method.Linear,
          Method.Newton,
          Method.Lagrange,
          ...
      ]

      # create interpolator
      synch_stream = Synchronizer("cpu")
      interp = Interpolator(grid, data, methods, synch_stream)
      interp.build_coefficients(4);  # build coefficients with 4 threads in the background

      # interpolation
      points = Array(...)                      # points has shape [npoints, ndim] and must be C-contiguous
      # points = Parcel(...)                   # use GPU array if the synch_stream is on GPU
      result = interp.evaluate_cpu(points, 8)  # asynchronous interpolation using 8 threads
      synch_stream.synchronize()               # for the main thread to wait until all tasks finished

.. warning::
    At the moment the method ``evaluate`` is invoked, the ``result`` is **NOT** ready. The result is only available
    **after** ``synch_stream`` **is synchronized**. This applied for both CPU and GPU interpolation.

Regression
^^^^^^^^^^

Polynomial regression generalizes interpolation by favoring overall error over the exact evaluation at all points on the
provided grid. Merlin also provides a library named ``regpl`` for polynomial regression using monomials. The utilization
is similar to ``splint``.

.. tab-set-code::

   .. code-block:: c++

      #include "merlin/array/array.hpp"          // merlin::array::Array
      #include "merlin/config.hpp"               // merlin::Index, merlin::make_array
      #include "merlin/grid/cartesian_grid.hpp"  // merlin::grid::CartesianGrid
      #include "merlin/regpl/polynomial.hpp"     // merlin::regpl::Polynomial
      #include "merlin/regpl/regressor.hpp"      // merlin::regpl::Regressor
      #include "merlin/regpl/vandermonde.hpp"    // merlin::regpl::Vandermonde
      #include "merlin/synchronizer.hpp"         // merlin::Synchronizer, merlin::ProcessorType
      #include "merlin/vector.hpp"               // merlin::DoubleVec

      merlin::array::Array data = ...;         // initialize data (C-contiguous array!)
      merlin::grid::CartesianGrid grid = ...;  // initialize grid

      // QR decomposition of the Vandermonde matrix
      merlin::regpl::Vandermonde coeff_calc(
          merlin::make_array({2, 3, 5, 10, ...}),  // order per axis of the polynomial
          grid,                                    // grid of points to fit
          4,                                       // number of CPU threads to process
      );

      // calculate polynomial coefficients and evaluate in parallel
      merlin::Synchronizer synch_stream(merlin::ProcessorType::Cpu);
      merlin::regpl::Regressor reg_poly(merlin::regpl::Polynomial(), synch_stream);
      coeff_calc.solve(data.data(), reg_poly.polynom());  // calculate coefficients
      reg_poly.evaluate(points, results, 8);              // evaluate (similar to Interpolator)
      synch_stream.synchronize();

   .. code-block:: python

      from merlin import Synchronizer
      from merlin.array import Array
      from merlin.grid import CartesianGrid
      from merlin.regpl import Polynomial, Regressor, create_vandermonde
      
      data = Array(...)          # initialize 1d of ravelled data from C-contiguous data
      grid = CartesianGrid(...)  # initialize grid

      # QR decomposition of the Vandermonde matrix
      coeff_calc = create_vandermonde(
          [2, 3, 5, 10, ...],  # order per axis of the polynomial
          grid,                # grid of points to fit
          4,                   # number of CPU threads to process
      )

      # calculate polynomial coefficients and evaluate in parallel
      synch_stream = Synchronizer("cpu")
      polynom = coeff_calc.solve(data)  # calculate coefficients
      reg_poly = Regressor(polynom, synch_stream)
      result = reg_poly.evaluate_cpu(points, 8)  # evaluate (similar to Interpolator)
      synch_stream.synchronize()
