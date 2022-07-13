namespace merlin {

// Constructor from pointer to data
template <typename Scalar>
Array<Scalar>::Array(Scalar * data, unsigned int ndim, unsigned int * dims,
             unsigned int * strides, bool copy) {
    // copy meta data
    this->ndim_ = ndim;
    this->dims_ = std::vector<unsigned int>(dims, dims + ndim);
    this->strides_ = std::vector<unsigned int>(strides, strides + ndim);
    this->is_copy = copy;

    // create begin and end iterator
    this->begin_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_ = std::vector<unsigned int>(this->ndim_, 0);
    this->end_[0] = this->dims_[0];

    // longest contiguous segment and break index
    std::vector<unsigned int> contiguous_strides(this->ndim_);
    contiguous_strides[this->ndim_-1] = sizeof(Scalar);
    for (int i = ndim-2; i >= 0; i--) {
        contiguous_strides[i] = contiguous_strides[i+1] * this->dims_[i+1];
    }
    this->longest_contiguous_segment_ = sizeof(Scalar);
    this->break_index_ = ndim - 1;
    for (int i = ndim-1; i >= 0; i--) {
        if (this->strides_[i] == contiguous_strides[i]) {
            this->longest_contiguous_segment_ *= dims[i];
            this->break_index_--;
        } else {
            break;
        }
    }

    // copy / assign data
    if (is_copy) {  // copy data
        // allocate a new array
        this->data_ = new Scalar [this->size()];

        // reform the stride array (force into C shape)
        this->strides_ = contiguous_strides;

        // copy data from old array to new array (optimized with memcpy)
        if (this->break_index_ == -1) {  // original array is perfectly contiguous
            std::memcpy(this->data_, data, this->longest_contiguous_segment_);
        } else {  // memcpy each longest_contiguous_segment
            for (Array::iterator it = this->begin(); it != this->end();) {
                unsigned int leap = 0;
                for (int i = 0; i < it.index().size(); i++) {
                    leap += it.index()[i] * strides[i];
                }
                uintptr_t src_ptr = (uintptr_t) data + leap;
                uintptr_t des_ptr = (uintptr_t) &(this->operator[](it.index()));
                std::memcpy((Scalar *) des_ptr, (Scalar *) src_ptr,
                            this->longest_contiguous_segment_);
                it.index()[this->break_index_] += 1;
                try {
                    it.update();
                } catch (const std::out_of_range & err) {
                    break;
                }
            }
        }

        // reset longest contiguous segment and break index
        this->longest_contiguous_segment_ = sizeof(Scalar);
        for (int i = ndim-1; i >= 0; i--) {
            this->longest_contiguous_segment_ *= dims[i];
        }
        this->break_index_ = -1;
    } else {
        this->data_ = data;
    }
}

// get item operator
template <typename Scalar>
Scalar & Array<Scalar>::operator[] (const std::vector<unsigned int> & index) {
    unsigned int leap = 0;
    if (index.size() != this->ndim_) {
        throw std::length_error("Index must have the same length as array");
    }
    for (int i = 0; i < index.size(); i++) {
        leap += index[i] * this->strides_[i];
    }
    uintptr_t data_ptr = (uintptr_t) this->data_ + leap;
    return *((Scalar *) data_ptr);
}

// pre-increment iterator
template <typename Scalar>
typename Array<Scalar>::iterator& Array<Scalar>::iterator::operator++(void) {
    this->index_[this->index_.size()-1]++;
    unsigned int current_dim = this->index_.size()-1;
    std::vector<unsigned int> & dims = this->dims();
    while (this->index_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == dims[current_dim]) {
                break;
            }
            else {
                throw(std::out_of_range("Maximum size reached, cannot add more."));
            }
        }
        this->index_[current_dim] = 0;
        this->index_[--current_dim] += 1;
    }
    return *this;
}

// post-increment operator
template <typename Scalar>
typename Array<Scalar>::iterator Array<Scalar>::iterator::operator++(int) {
    return ++(*this);
}

// destructor
template <typename Scalar>
Array<Scalar>::~Array(void) {
    if (this->is_copy) {
        std::printf("Free copied data.\n");
        delete[] this->data_;
    }
    #ifdef __NVCC__
    if (this->gpu_data_ != NULL) {
        std::printf("Free GPU data.\n");
        cudaFree(this->gpu_data_);
    }
    #endif
}

// get size
template <typename Scalar>
unsigned int Array<Scalar>::size(void) {
    unsigned int size = 1;
    for (int i = 0; i < this->ndim_; i++) {
        size *= this->dims_[i];
    }
    return size;
}

// begin iterator
template <typename Scalar>
typename Array<Scalar>::iterator Array<Scalar>::begin(void) {
    return Array<Scalar>::iterator(this->begin_, this->dims_);
}

// end iterator
template <typename Scalar>
typename Array<Scalar>::iterator Array<Scalar>::end(void) {
    return Array<Scalar>::iterator(this->end_, this->dims_);
}

template <typename Scalar>
void Array<Scalar>::iterator::update(void) {
    // detect dimensions having index bigger than dim
    unsigned int current_dim = this->index_.size();
    std::vector<unsigned int> & dims = this->dims();
    for (int i = this->index_.size() - 1; i >= 0; i--) {
        if (this->index_[i] >= dims[i]) {
            current_dim = i;
            break;
        }
    }
    if (current_dim == this->index_.size()) {  // no update needed
        return;
    }

    // carry the surplus to the dimensions with bigger strides
    while (this->index_[current_dim] >= dims[current_dim]) {
        if (current_dim == 0) {
            if (this->index_[current_dim] == dims[current_dim]) {
                break;
            }
            else {
                throw(std::out_of_range("Maximum size reached, cannot add more."));
            }
        }
        div_t carry = div((int) this->index_[current_dim], (int) dims[current_dim]);
        this->index_[current_dim] = carry.rem;
        this->index_[--current_dim] += carry.quot;
    }
}

#ifdef __NVCC__
// copy data to GPU
template <typename Scalar>
void Array<Scalar>::sync_to_gpu(cudaStream_t stream = NULL) {
    // allocate data on GPU
    if (this->gpu_data_ == NULL) {
        cudaMalloc(&(this->gpu_data_), sizeof(Scalar)*this->size());
    }

    // GPU stride array
    std::vector<unsigned int> gpu_strides_(this->ndim_, 0);
    gpu_strides_[this->ndim_-1] = sizeof(Scalar);
    for (int i = this->ndim_-2; i >= 0; i--) {
        gpu_strides_[i] = gpu_strides_[i+1] * this->dims_[i+1];
    }

    // copy data to GPU
    if (this->break_index_ == -1) {  // original array is perfectly contiguous
        cudaMemcpy(this->gpu_data_, this->data_,
                   this->longest_contiguous_segment_, cudaMemcpyHostToDevice);
    } else {  // copy each longest_contiguous_segment
        for (Array::iterator it = this->begin(); it != this->end();) {
            // cpu leap
            unsigned int cpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                cpu_leap += it.index()[i] * this->strides_[i];
            }

            // gpu leap
            unsigned int gpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                gpu_leap += it.index()[i] * gpu_strides_[i];
            }

            // clone data
            uintptr_t src_ptr = (uintptr_t) this->data_ + cpu_leap;
            uintptr_t des_ptr = (uintptr_t) this->gpu_data_ + gpu_leap;
            cudaMemcpy((Scalar *) des_ptr, (Scalar *) src_ptr,
                        this->longest_contiguous_segment_, cudaMemcpyHostToDevice);

            // increment iterator
            it.index()[this->break_index_] += 1;
            try {
                it.update();
            } catch (const std::out_of_range & err) {
                break;
            }
        }
    }
}

template <typename Scalar>
void Array<Scalar>::sync_from_gpu(cudaStream_t stream = NULL) {
    // GPU stride array
    std::vector<unsigned int> gpu_strides_(this->ndim_, 0);
    gpu_strides_[this->ndim_-1] = sizeof(Scalar);
    for (int i = this->ndim_-2; i >= 0; i--) {
        gpu_strides_[i] = gpu_strides_[i+1] * this->dims_[i+1];
    }

    // copy data from GPU
    if (this->break_index_ == -1) {  // original array is perfectly contiguous
        cudaMemcpy(this->data_, this->gpu_data_,
                   this->longest_contiguous_segment_, cudaMemcpyDeviceToHost);
    } else {  // copy each longest_contiguous_segment
        for (Array::iterator it = this->begin(); it != this->end();) {
            // cpu leap
            unsigned int cpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                cpu_leap += it.index()[i] * this->strides_[i];
            }

            // gpu leap
            unsigned int gpu_leap = 0;
            for (int i = 0; i < it.index().size(); i++) {
                gpu_leap += it.index()[i] * gpu_strides_[i];
            }

            // clone data
            uintptr_t src_ptr = (uintptr_t) this->gpu_data_ + gpu_leap;
            uintptr_t des_ptr = (uintptr_t) this->data_ + cpu_leap;
            cudaMemcpy((Scalar *) des_ptr, (Scalar *) src_ptr,
                        this->longest_contiguous_segment_, cudaMemcpyDeviceToHost);

            // increment iterator
            it.index()[this->break_index_] += 1;
            try {
                it.update();
            } catch (const std::out_of_range & err) {
                break;
            }
        }
    }
}
#endif

}  // namespace merlin