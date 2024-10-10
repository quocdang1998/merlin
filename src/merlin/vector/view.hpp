// Copyright 2024 quocdang1998
#ifndef MERLIN_VECTOR_VIEW_HPP_
#define MERLIN_VECTOR_VIEW_HPP_

#include <concepts>  // std::convertible_to
#include <cstddef>   // nullptr
#include <cstdint>   // std::uint64_t
#include <iterator>  // std::contiguous_iterator, std::distance, std::iter_reference_t
#include <memory>    // std::to_address
#include <ranges>    // std::ranges::equal
#include <string>    // std::string

#include "merlin/config.hpp"                   // __cuhostdev__, __cudevice__
#include "merlin/vector/declaration.hpp"       // merlin::vector::View
#include "merlin/vector/iterator_helpers.hpp"  // merlin::vector::ForwardIterator, merlin::vector::ReverseIterator

namespace merlin {

// Range view
// ----------

/** @brief Provide a view to a constant contiguous array.
 *  @note Undefined behavior if the assigned data is de-allocated prior to the destruction of the object.
 *  @tparam T Element type.
 */
template <typename T>
class vector::View {
  public:
    /// @name Constructors
    /// @{
    /** @brief Default constructor.*/
    __cuhostdev__ constexpr View(void) {}
    /** @brief Constructor from a range pointer and the number of elements in the range.
     *  @tparam Pointer Pointer type. Must be a contiguous iterator and its dereference must be convertible to ``T``.
     */
    template <typename Pointer>
    requires std::contiguous_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const T &>
    __cuhostdev__ constexpr View(Pointer data, std::uint64_t size) : data_(std::to_address(data)), size_(size) {}
    /** @brief Constructor from pointer to the first and the last iterator.
     *  @tparam Pointer Pointer type. Must be a contiguous iterator and its dereference must be convertible to ``T``.
     */
    template <typename Pointer>
    requires std::contiguous_iterator<Pointer> && std::convertible_to<std::iter_reference_t<Pointer>, const T &>
    __cuhostdev__ constexpr View(Pointer first, Pointer last) :
    data_(std::to_address(first)), size_(std::distance(first, last)) {}
    /// @}

    /// @name Get members
    /// @{
    /** @brief Get constant reference to pointer of data.*/
    __cuhostdev__ constexpr const T * data(void) const noexcept { return this->data_; }
    /** @brief Get constant reference to size.*/
    __cuhostdev__ constexpr const std::uint64_t & size(void) const noexcept { return this->size_; }
    /// @}

    /// @name Get element
    /// @{
    /** @brief Get constant reference to an element.*/
    __cuhostdev__ constexpr const T & operator[](std::uint64_t i) const noexcept { return this->data_[i]; }
    /// @}

    /// @name Forward iterators
    /// @{
    /** @brief Forward iterator type.*/
    using iterator = vector::ForwardIterator<const T>;
    /** @brief Begin iterator.*/
    __cuhostdev__ constexpr iterator begin(void) const { return iterator(this->data_); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr iterator cbegin(void) const { return iterator(this->data_); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr iterator end(void) const { return iterator(this->data_ + this->size_); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr iterator cend(void) const { return iterator(this->data_ + this->size_); }
    /// @}

    /// @name Reverse iterators
    /// @{
    /** @brief Reverse iterator type.*/
    using reverse_iterator = vector::ReverseIterator<const T>;
    /** @brief Reverse begin iterator.*/
    __cuhostdev__ constexpr reverse_iterator rbegin(void) const { return reverse_iterator(this->data_ + this->size_); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr reverse_iterator crbegin(void) const { return reverse_iterator(this->data_ + this->size_); }
    /** @brief End iterator.*/
    __cuhostdev__ constexpr reverse_iterator rend(void) const { return reverse_iterator(this->data_); }
    /** @brief Constant begin iterator.*/
    __cuhostdev__ constexpr reverse_iterator crend(void) const { return reverse_iterator(this->data_); }
    /// @}

    /// @name Subspan
    /// @{
    /** @brief Get a view to the first few elements.
     *  @note Undefined behavior if the number of elements exceed the size of the view.
     */
    __cuhostdev__ constexpr vector::View<T> get_head(std::uint64_t n) {
        return vector::View<T>(this->data_, n);
    }
    /** @brief Get a view to the last few elements.
     *  @note Undefined behavior if the number of elements exceed the size of the view.
     */
    __cuhostdev__ constexpr vector::View<T> get_tail(std::uint64_t n) {
        return vector::View<T>(this->data_ + this->size_ - n, this->data_ + this->size_);
    }
    /** @brief Create a subspan from a view.*/
    __cuhostdev__ constexpr vector::View<T> get_subspan(std::uint64_t offset, std::uint64_t length) const {
        return vector::View<T>(this->data_ + offset, length);
    }
    /// @}

    /// @name Comparison
    /// @{
    friend __cuhostdev__ constexpr bool operator==(const vector::View<T> & v1, const vector::View<T> & v2) {
        if (v1.size_ != v2.size_) {
            return false;
        }
        return std::ranges::equal(v1, v2);
    }
    friend __cuhostdev__ constexpr bool operator!=(const vector::View<T> & v1, const vector::View<T> & v2) {
        return !(v1 == v2);
    }
    /// @}

    /// @name String representation
    /// @{
    /** @brief String representation.
     *  @param sep Separator between printed elements.
     */
    std::string str(const char * sep = " ") const;
    /// @}

  private:
    /** @brief Pointer to data.*/
    const T * data_ = nullptr;
    /** @brief Number of element of the range.*/
    std::uint64_t size_ = 0;
};

}  // namespace merlin

#include "merlin/vector/view.tpp"

#endif  // MERLIN_VECTOR_VIEW_HPP_
