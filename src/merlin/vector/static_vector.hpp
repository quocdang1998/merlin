// Copyright 2024 quocdang1998
#ifndef MERLIN_VECTOR_STATIC_VECTOR_HPP_
#define MERLIN_VECTOR_STATIC_VECTOR_HPP_

#include <array>             // std::array
#include <concepts>          // std::convertible_to
#include <cstdint>           // std::uint64_t
#include <initializer_list>  // std::initializer_list, std::data
#include <iterator>          // std::forward_iterator, std::distance, std::iter_reference_t
#include <string>            // std::string

#include "merlin/config.hpp"                   // __cuhostdev__, __cudevice__
#include "merlin/vector/declaration.hpp"       // merlin::vector::StaticVector
#include "merlin/vector/iterator_helpers.hpp"  // merlin::vector::ForwardIterator, merlin::vector::ReverseIterator
#include "merlin/vector/view.hpp"              // merlin::vector::View

namespace merlin {

// Inplace vector
// --------------

/** @brief Inplace vector with a max size.*/
template <class T, std::uint64_t Capacity>
class vector::StaticVector {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ constexpr StaticVector(void) {}
    /** @brief Constructor from initializer list.*/
    __cuhostdev__ constexpr StaticVector(std::initializer_list<T> data) {
        this->size_ = ((Capacity > data.size()) ? data.size() : Capacity);
        for (std::uint64_t i = 0; i < this->size_; i++) {
            this->data_holder_[i] = std::data(data)[i];
        }
    }
    /** @brief Constructor from size and fill-in value.
     *  @param size Number of element.
     *  @param value Value of each element.
     */
    __cuhostdev__ constexpr StaticVector(std::uint64_t size, const T & value = T()) {
        this->size_ = ((Capacity > size) ? size : Capacity);
        this->data_holder_.fill(value);
    }
    /** @brief Constructor from pointer to an iterator and size.
     *  @details Data are copied from the source location into the vector upto the max capacity.
     *  @param data Pointer to the first element of source array.
     *  @param size Size of resulted vector (can be smaller or equals the original array).
     */
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const T &>
    __cuhostdev__ constexpr StaticVector(Pointer data, std::uint64_t size) {
        this->size_ = ((Capacity > size) ? size : Capacity);
        for (std::uint64_t i = 0; i < this->size_; i++) {
            new (this->data_holder_.data() + i) T(*data);
            ++data;
        }
    }
    /** @brief Constructor from a range.
     *  @details Data are copied from the range into the vector upto the max capacity.
     *  @param first Pointer to the first element.
     *  @param last Pointer to the last element.
     */
    template <typename Pointer>
    requires std::forward_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const T &>
    __cuhostdev__ constexpr StaticVector(Pointer first, Pointer last) {
        std::uint64_t distance = std::distance(first, last);
        this->size_ = ((Capacity > distance) ? distance : Capacity);
        for (std::uint64_t i = 0; first != last && i < this->size_; i++) {
            new (this->data_holder_.data() + i) T(*first);
            ++first;
        }
    }
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get reference to pointer of data.*/
    __cuhostdev__ constexpr T * data(void) { return this->data_holder_.data(); }
    /** @brief Get constant reference to pointer of data.*/
    __cuhostdev__ constexpr const T * data(void) const noexcept { return this->data_holder_.data(); }
    /** @brief Get constant reference to size.*/
    __cuhostdev__ constexpr const std::uint64_t & size(void) const noexcept { return this->size_; }
    /// @}

    /// @name Get element
    /// @{
    /** @brief Get reference to an element.*/
    __cuhostdev__ constexpr T & operator[](std::uint64_t i) noexcept { return this->data_holder_[i]; }
    /** @brief Get constant reference to an element.*/
    __cuhostdev__ constexpr const T & operator[](std::uint64_t i) const noexcept { return this->data_holder_[i]; }
    /// @}

    /// @name Resize
    /// @{
    /** @brief Resize vector.
     *  @details If the new size is greater than the capacity, the new size will be truncated to the capacity.
     */
    __cuhostdev__ constexpr void resize(std::uint64_t new_size) noexcept {
        this->size_ = ((Capacity > new_size) ? new_size : Capacity);
    }
    /// @}

    /// @name Forward iterators
    /// @{
    /** @brief Forward iterator type.*/
    using iterator = vector::ForwardIterator<T>;
    /** @brief Constant forward iterator type.*/
    using const_iterator = vector::ForwardIterator<const T>;
    /** @brief Begin iterator.*/
    __cuhostdev__ constexpr iterator begin(void) { return iterator(this->data_holder_.data()); }
    /** @brief Begin iterator.*/
    __cuhostdev__ constexpr const_iterator begin(void) const { return const_iterator(this->data_holder_.data()); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_iterator cbegin(void) const { return const_iterator(this->data_holder_.data()); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr iterator end(void) { return iterator(this->data_holder_.data() + this->size_); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr const_iterator end(void) const {
        return const_iterator(this->data_holder_.data() + this->size_);
    }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_iterator cend(void) const {
        return const_iterator(this->data_holder_.data() + this->size_);
    }
    /// @}

    /// @name Reverse iterators
    /// @{
    /** @brief Reverse iterator type.*/
    using reverse_iterator = vector::ReverseIterator<T>;
    /** @brief Constant reverse iterator type.*/
    using const_reverse_iterator = vector::ReverseIterator<const T>;
    /** @brief Reverse begin iterator.*/
    __cuhostdev__ constexpr reverse_iterator rbegin(void) {
        return reverse_iterator(this->data_holder_.data() + this->size_);
    }
    /** @brief Reverse begin iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator rbegin(void) const {
        return const_reverse_iterator(this->data_holder_.data() + this->size_);
    }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator crbegin(void) const {
        return const_reverse_iterator(this->data_holder_.data() + this->size_);
    }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr reverse_iterator rend(void) { return reverse_iterator(this->data_holder_.data()); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator rend(void) const {
        return const_reverse_iterator(this->data_holder_.data());
    }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr const_reverse_iterator crend(void) const {
        return const_reverse_iterator(this->data_holder_.data());
    }
    /// @}

    /// @name Get view
    /// @{
    /** @brief Get a view corresponding to the vector.*/
    __cuhostdev__ constexpr vector::View<T> get_view(void) const {
        return vector::View<T>(this->data_holder_.data(), this->size_);
    }
    /// @}

    /// @name Comparison
    /// @{
    friend __cuhostdev__ constexpr bool operator==(const vector::StaticVector<T, Capacity> & v1,
                                                   const vector::StaticVector<T, Capacity> & v2) {
        return v1.get_view() == v2.get_view();
    }
    friend __cuhostdev__ constexpr bool operator!=(const vector::StaticVector<T, Capacity> & v1,
                                                   const vector::StaticVector<T, Capacity> & v2) {
        return v1.get_view() != v2.get_view();
    }
    /// @}

    /// @name String representation
    /// @{
    /** @brief String representation.
     *  @param sep Separator between printed elements.
     */
    std::string str(const char * sep = " ") const { return this->get_view().str(sep); }
    /// @}

  private:
    /** @brief Data container.*/
    std::array<T, Capacity> data_holder_;
    /** @brief Number of elements.*/
    std::uint64_t size_ = 0;
};

}  // namespace merlin

#endif  // MERLIN_VECTOR_STATIC_VECTOR_HPP_
