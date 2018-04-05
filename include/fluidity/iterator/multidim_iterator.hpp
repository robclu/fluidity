//==--- fluidity/iterator/multidim_iterator.hpp ------------ -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multidim_iteartor.hpp
/// \brief This file defines the implementation of an iterator for a multi
///        dimensional space.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_HPP
#define FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_HPP

#include <fluidity/dimension/dimension_info.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid {

/// The MultidimIterator class defines an iterator class which can iterate over
/// the different dimensions in a multi dimensional space.
/// \tparam T         The type of the data to iterate over.
/// \tparam DimInfo   Information for the dimensions.
template <typename T, typename DimInfo>
struct MultidimIterator {
  /// Defines the type of this iterator.
  using self_t          = MultidimIterator;
  /// Defines the type of the data being iterated over.
  using value_t         = std::decay_t<T>;
  /// Defines the type of the pointer for the iterator.
  using pointer_t       = value_t*;
  /// Defines the type of a constant pointer for the iterator.
  using const_pointer_t = const value_t*;
  /// Defines the type of the information for the dimensions.
  using dim_info_t      = std::decay_t<DimInfo>;

  /// Defines the number of dimensions for the iterator.
  static constexpr std::size_t dimensions = dim_info_t::num_dimensions();

 private:
  pointer_t _ptr; //!< A pointer to the data to iterate over.

  /// Defines the default dimension, so that the iterator behaves like a
  /// ContiguousIterator in the base case.
  static constexpr auto default_dim = dim_x;

 public:
  /// Initializes the pointer to data to iterate over.
  /// \param[in] ptr The pointer to iterate from.
  fluidity_host_device MultidimIterator(pointer_t ptr) : _ptr(ptr)
  {
    assert(_ptr != nullptr && "Cannot iterate from a nullptr!");
  }

  /// Returns amount of offset required to iterate in dimension \p dim. The
  /// offset in the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value> 
  fluidity_host_device static constexpr std::size_t
  offset(Dimension<Value> dim = default_dim)
  {
    return dim_info_t::offset(Dimension<Value>{});
  }

  /// Overload of operator() to offset the iterator in a specific dimension.
  /// \param[in] amount The amount (number of elements) to iterate over.
  /// \param[in] dim    The dimension to iterator over.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device constexpr auto
  operator()(int amount, Dimension<Value> dim = default_dim) const
  {
    return self_t{_ptr + amount * offset(Dimension<Value>{})};
  }

  /// Overload of operator* to get the value the iterator points to.
  fluidity_host_device constexpr value_t&  operator*()
  {
    return *_ptr;
  }

  /// Overload of operator* to get the a const reference to the value the
  /// iterator points to.
  fluidity_host_device constexpr const value_t& operator*() const
  {
    return *_ptr;
  }

  /// Overload of operator-> to access the value the iterator points to.
  fluidity_host_device constexpr pointer_t operator->()
  {
    return _ptr;
  }

  /// Overload of operator-> to access the value the iterator points to as const.
  fluidity_host_device constexpr const_pointer_t operator->() const
  {
    return _ptr;
  }

  /// Offsets the iterator by \p amount in the dimension defined by \p dim, and
  /// returns a new iterator to the offset element.
  /// \param[in]  amount  The amount of offset from this iterator.
  /// \param[in]  dim     The dimension in which to offset.
  /// \tparam     Value   The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device constexpr self_t
  offset(int amount, Dimension<Value> dim = default_dim) const
  {
    return self_t{_ptr + amount * offset(Dimension<Value>{})};
  }
  
  /// Shifts the iterator by \p amount in dimension \p dim, modifying this
  /// iterator. This can shift the iterator forward and backward in the given
  /// dimension based on the sign of \p amount (+ = forward).
  /// \param[in]  amount  The amount to advance the iterator by.
  /// \param[in]  dim     The dimension to advance the iterator in.
  /// \tparam     Value   The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device constexpr self_t&
  shift(int amount, Dimension<Value> /*dim*/)
  {
    _ptr += amount * offset(Dimension<Value>{});
    return *this;
  }

  /// Returns the backward difference between this iterator and the iterator \p
  /// amount places from from this iterator in dimension \p dim. I.e
  /// 
  /// \begin{equation}
  ///   \Delta U = U_\textrm{dim}(i) - U_{\textrm{dim}(i - \texrm{amount})}
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state->backward_diff(fluid::dim_x);
  /// auto diff = state->backward_diff(fluid::dim_x, 1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     Value   The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device constexpr value_t
  backward_diff(Dimension<Value> /*dim*/, unsigned int amount = 1) const
  {
    return *_ptr - *(_ptr - amount * offset(Dimension<Value>{}));
  }

  /// Returns the forward difference between this iterator and the iterator \p
  /// amount places from from this iterator in dimension \p dim. I.e
  /// 
  /// \begin{equation}
  ///   \Delta U = U_{\textrm{dim}(i + \texrm{amount})} - U_\textrm{dim}(i)
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state->forward_diff(fluid::dim_y);
  /// auto diff = state->forward_diff(fluid::dim_y, 1);
  /// 
  /// // But is different from (as this returns the forward difference between
  /// // the element 2 positions ahead of the iterator's element and the
  /// // iterator's element):
  /// auto diff = state->forward_diff(fluid::dim_y, 2);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     Value   The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device constexpr value_t
  forward_diff(Dimension<Value> /*dim*/, unsigned int amount = 1) const
  {
    return *(_ptr + amount * offset(Dimension<Value>{})) - *_ptr;
  }

  /// Returns the central difference between this iterator and the iterators \p
  /// amount places forward and backward from this iterator in dimension \p dim.
  /// 
  /// \begin{equation}
  ///   \Delta U = U_{\textrm{dim}(i + \texrm{amount})} 
  ///            - U_{\textrm{dim}(i - \texrm{amount})} 
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state->centralDiff(fluid::dim_z);
  /// auto diff = state->centralDiff(fluid::dim_z, 1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     Value   The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device constexpr value_t
  central_diff(Dimension<Value> /*dim*/, unsigned int amount = 1) const
  {
    constexpr auto dim = Dimension<Value>{};
    return *(_ptr + amount * offset(dim)) - *(_ptr - amount * offset(dim));
  }
};

#if defined(__CUDACC__)

/// Makes a multidimensional iterator over a multidimensional space, where the
/// properties of the space are defined by the DimInfo parameter. The DimInfo
/// template parameter must be of DimInfoCt type, otherwise a compiler error is
/// generated.
/// \tparam T       The type of the data to iterate over.
/// \tparam DimInfo The information which defines the multi dimensional space.
template <typename T, typename DimInfo>
fluidity_device_only constexpr auto make_multidim_iterator()
{
  using iter_t = MultidimIterator<T, DimInfo>;
  __shared__ T buffer[DimInfo::total_size()];
  iter_t iter{buffer};

  // Move the iterator to the current thread.
  iter.shift(dim_x, thread_id(dim_x))
      .shift(dim_y, thread_id(dim_y))
      .shift(dim_z, thread_id(dim_z));
  return iter;
}

/// Makes a multidimensional iterator over a multidimensional space, where the
/// properties of the space are defined by the DimInfo parameter. The DimInfo
/// template parameter must be of DimInfoCt type, otherwise a compiler error is
/// generated.
/// \param[in] ptr     A pointer to the start of data to iterate over.      
/// \tparam    T       The type of the data to iterate over.
/// \tparam    DimInfo The information which defines the multi dimensional space.
template <typename T, typename DimInfo>
fluidity_device_only constexpr auto make_multidim_iterator(T* ptr)
{
  using iter_t = MultidimIterator<T, DimInfo>;
  iter_t iter{ptr};

  // Move the iterator to the current thread.
  iter.shift(dim_x, thread_id(dim_x))
      .shift(dim_y, thread_id(dim_y))
      .shift(dim_z, thread_id(dim_z));
  return iter;
}

#else

/// Makes a multidimensional iterator over a multidimensional space, where the
/// properties of the space are defined by the DimInfo parameter. The DimInfo
/// template parameter must be of DimInfoCt type, otherwise a compiler error is
/// generated.
/// \tparam T       The type of the data to iterate over.
/// \tparam DimInfo The information which defines the multi dimensional space.
template <typename T, typename DimInfo>
fluidity_host_only auto make_multidim_iterator()
{
  using iter_t = MultidimIterator<T, DimInfo>;
  static_assert(std::is_same_v<DimInfo, DimInfoCt>,
                "DimInfo must be DimInfoCt type to make a multidim iterator!");
  static thread_local T buffer[DimInfo::total_size()];
  iter_t iter{buffer};

  // Move the iterator to the current thread.
  iter.shift(dim_x, thread_id(dim_x))
      .shift(dim_y, thread_id(dim_y))
      .shift(dim_z, thread_id(dim_z));
  return iter;
}

#endif // __CUDACC__

} // namespace fluid

#endif // FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_HPP