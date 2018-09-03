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
/// \file  multidim_iterator.hpp
/// \brief This file defines the implementation of an iterator for a multi
///        dimensional space, which can have either runtime or compile time
///        offsets.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_HPP
#define FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_HPP

#include "strided_iterator.hpp"
#include <fluidity/dimension/dimension_info.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid {

/// The MultidimIterator class defines an iterator class which can iterate over
/// the different dimensions in a multi dimensional space, where the properties
/// of the space are defined by the DimInfo type.
/// 
/// Providing a DimInfo type which has constexpr offsets, all ofsetting can be
/// done at compile time, while if the DimInfo type is runtime, then the offets
/// are coputed at runtime and can be changed.
/// 
/// \tparam T         The type of the data to iterate over.
/// \tparam DimInfo   Information for the dimensions.
/// \tparam Exec      The execution policy for the iterator.
template < typename T
         , typename DimensionInfo = DimInfo<2>
         , typename Exec          = exec::default_type>
struct MultidimIterator : public DimensionInfo {
  /// Defines the type of this iterator.
  using self_t          = MultidimIterator;
  /// Defines the type of the data being iterated over.
  using value_t         = std::decay_t<T>;
  /// Defines the type of the pointer for the iterator.
  using pointer_t       = value_t*;
  /// Defines the type of a constant pointer for the iterator.
  using const_pointer_t = const value_t*;
  /// Defines the type of the information for the dimensions.
  using dim_info_t      = DimensionInfo;
  /// Defines the type of the execution policy for the iterator.
  using exec_t          = Exec;

  /// Defines if the strides are constexpr computed.
  static constexpr auto constexpr_strides    = dim_info_t::constexpr_offsets;
  /// Defines the number of dimensions for the iterator.
  static constexpr auto dimensions           = dim_info_t::dimensions;
  /// Defines that the iterator is multi-dimensional.
  static constexpr auto is_multi_dimensional = true;

 private:
  pointer_t _ptr; //!< A pointer to the data to iterate over.

  /// Defines the type of a non-const strided iterator to create.
  using strided_iter_t       = StridedIterator<value_t, false, exec_t>;
  /// Defines the type of a const strided iterator to create.
  using const_strided_iter_t = StridedIterator<value_t, true, exec_t>;
  
 public:
  /// Initializes the pointer to data to iterate over.
  /// \param[in] ptr The pointer to iterate from.
  fluidity_host_device MultidimIterator(pointer_t ptr) : _ptr{ptr}
  {
    //static_assert(constexpr_strides,
    //              "Can't use runtime offset iterator without the offsets!");
    assert(_ptr != nullptr && "Cannot iterate from a nullptr!");
  }

  /// Initializes the pointer to data to iterate over, as well as the sizes of
  /// the dimensions for the iterator.
  /// \param[in] ptr   The pointer to iterate from.
  /// \param[in] s1  The size of the first dimension to iterator over.
  /// \param[in] so  The sizes of the rest of the dimensions to iterate over.
  /// \tparam    S1  The type of the firs dimensions's size.
  /// \tparam    SO  The types of the rest of the dimension sizes.
  template <typename S1, typename... SO, var_or_int_enable_t<S1, SO...> = 0>
  fluidity_host_device
  MultidimIterator(pointer_t ptr, S1&& s1, SO&&... so)
  : dim_info_t{std::forward<S1>(s1), std::forward<SO>(so)...}, _ptr{ptr}
  {
    assert(_ptr != nullptr && "Cannot iterate from a nullptr!");
  }

  /// Initializes the pointer to data to iterate over, as well as the dimenison
  /// information.
  /// \param[in] ptr   The pointer to iterate from.
  /// \param[in] sizes The sizes of the dimensions to iterate over.
  /// \tparam    Sizes The types of the sizes.
  fluidity_host_device
  MultidimIterator(pointer_t ptr, const dim_info_t& dim_info)
  : dim_info_t{dim_info}, _ptr{ptr}
  {
    // NOTE: This assertation causes a problem ...
    //assert(_ptr != nullptr && "Cannot iterate from a nullptr!");
  }

  /// Overload of operator() to offset the iterator in a specific dimension.
  /// \param[in] amount The amount (number of elements) to iterate over.
  /// \param[in] dim    The dimension to iterator over.
  /// \tparam    Value  The value which defines the dimension.
/*
  template <std::size_t Value>
  fluidity_host_device constexpr auto
  operator()(int amount, Dimension<Value> dim) const
  {
    return self_t{_ptr + amount * offset(Dimension<Value>{}), *this};
  }
*/
  template <typename Dim>
  fluidity_host_device constexpr auto
  operator()(int amount, Dim dim) const
  {
    return self_t{_ptr + amount * offset(dim), *this};
  }

  /// Overload of operator[] to access the data as if it is a 1D array. 
  /// \param[in] index The global index of the element to access
  fluidity_host_device constexpr value_t& operator[](std::size_t index)
  {
    return _ptr[index];
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

  /// Overload of operator-> to access the value the iterator points to.
  fluidity_host_device constexpr const_pointer_t operator->() const
  {
    return _ptr;
  }

  /// Gets a raw pointer to the data.
  fluidity_host_device constexpr pointer_t get_ptr()
  {
    return _ptr;
  }

  /// Gets a const raw pointer to the data.
  fluidity_host_device constexpr const_pointer_t get_ptr() const
  {
    return _ptr;
  }

  /// Returns a strided iterator which can iterate over the \p dim dimension.
  /// \param[in] dim   The dimension for the iterator to iterate over.
  /// \tparam    value The value which defines the dimension.
/*
  template <std::size_t Value>
  fluidity_host_device strided_iter_t
  as_strided_iterator(Dimension<Value> dim)
  {
    using stride_t = typename strided_iter_t::stride_t;
    return strided_iter_t{_ptr,
                          static_cast<stride_t>(stride(Dimension<Value>{}))};
  }
*/
  template <typename Dim>
  fluidity_host_device strided_iter_t as_strided_iterator(Dim dim)
  {
    using stride_t = typename strided_iter_t::stride_t;
    return strided_iter_t{_ptr, static_cast<stride_t>(stride(dim))};
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
  /// auto diff = state.backward_diff(fluid::dim_x);
  /// auto diff = state.backward_diff(fluid::dim_x, 1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     Value   The value which defines the dimension.
  template <typename Dim>
  fluidity_host_device constexpr value_t
  backward_diff(Dim dim, unsigned int amount = 1) const
  {
    return *_ptr - *(_ptr - amount * stride(dim));
  }

  /// Returns the backward difference between this iterator and the iterator \p
  /// amount places from from this iterator in dimension \p dim, for the Nth
  /// element of the data which the iterator points to. This method can only be
  /// used when the iterated data has overloaded operator[]. I.e
  /// 
  /// \begin{equation}
  ///   \Delta U = U_\textrm{dim}(i)[n] -
  ///            - U_{\textrm{dim}(i - \texrm{amount})}[n]
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state.backward_diff<N>(fluid::dim_x);
  /// auto diff = state.backward_diff<N>(fluid::dim_x, 1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     N       The index of the element to compute the difference of.
  /// \tparam     Value   The value which defines the dimension.
  /*
  template <std::size_t N, typename Dim>
  fluidity_host_device constexpr auto
  backward_diff(Dim dim, unsigned int amount = 1) const
  {
    return (*_ptr)[N] - (*(_ptr - amount * stride(dim)))[N];
  }
  */

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
  /// auto diff = state.centralDiff(fluid::dim_z);
  /// auto diff = state.centralDiff(fluid::dim_z, 1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     Value   The value which defines the dimension.
  template <typename Dim>
  fluidity_host_device constexpr value_t
  central_diff(Dim dim, unsigned int amount = 1) const
  {
    const auto shift = amount * stride(dim);
    return *(_ptr + shift) - *(_ptr - shift);
  }

  /// Returns the central difference between this iterator and the iterators \p
  /// amount places forward and backward from this iterator in dimension \p dim,
  /// for the Nth element of the iterated over data. This method can only be
  /// used when the iterated data overloads operator[].
  /// 
  /// \begin{equation}
  ///   \Delta U = U_{\textrm{dim}(i + \texrm{amount})}[N] 
  ///            - U_{\textrm{dim}(i - \texrm{amount})}[N]
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state.centralDiff<N>(fluid::dim_z);
  /// auto diff = state.centralDiff<N>(fluid::dim_z, 1);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     N       The index of the element to difference.
  /// \tparam     Value   The value which defines the dimension.
  /*
  template <std::size_t N, std::size_t Value>
  fluidity_host_device constexpr auto
  central_diff(unsigned int amount = 1) const
  {
    const auto shift = amount * stride(Dimension<Value>{});
    return (*(_ptr + shift))[N] - (*(_ptr - shift))[N];
  }
  */

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
  /// auto diff = state.forward_diff(fluid::dim_y);
  /// auto diff = state.forward_diff(fluid::dim_y, 1);
  /// 
  /// // But is different from (as this returns the forward difference between
  /// // the element 2 positions ahead of the iterator's element and the
  /// // iterator's element):
  /// auto diff = state.forward_diff(fluid::dim_y, 2);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \param[in]  dim     The dimension to offset in.
  /// \tparam     Value   The value which defines the dimension.
  template <typename Dim>
  fluidity_host_device constexpr value_t
  forward_diff(Dim dim, unsigned int amount = 1) const
  {
    return *(_ptr + amount * stride(dim)) - *_ptr;
  }

  /// Returns the forward difference between this iterator and the iterator \p
  /// amount places from from this iterator in dimension \p dim, for the Nth
  /// element in the data pointed to by the iterator. This function can only be
  /// used when the iterator iterates over data which overloads operator[]. The
  /// forward difference is:
  /// 
  /// \begin{equation}
  ///   \Delta U = U_{\textrm{dim}(i + \texrm{amount})}[N] 
  ///            - U_\textrm{dim}(i)[N]
  /// \end{equation}
  /// 
  /// The default is that \p amount is 1, i.e:
  /// 
  /// ~~~cpp
  /// // The following is the same:
  /// auto diff = state.forward_diff<N>(fluid::dim_y);
  /// auto diff = state.forward_diff<N>(fluid::dim_y, 1);
  /// 
  /// // But is different from (as this returns the forward difference between
  /// // the element 2 positions ahead of the iterator's element and the
  /// // iterator's element):
  /// auto diff = state.forward_diff<N>(fluid::dim_y, 2);
  /// ~~~
  /// 
  /// \param[in]  amount  The amount to offset the iterator by.
  /// \tparam     Value   The value which defines the dimension.
  /// \tparam     N       The index of the element to to difference.
  /// \tparam     Value   The value which defines the dimension.
  /*
  template <std::size_t N, std::size_t Value>
  fluidity_host_device constexpr auto
  forward_diff(unsigned int amount = 1) const
  {
    return (*(_ptr + amount * stride(Dimension<Value>{})))[N] - (*_ptr)[N];
  }
  */
 
  /// Returns the number of dimensions which can be iterated over.
  fluidity_host_device constexpr std::size_t num_dimensions() const
  {
    return dimensions;
  }

  /// Offsets the iterator by \p amount in the dimension defined by \p dim, and
  /// returns a new iterator to the offset element.
  /// \param[in]  amount  The amount of offset from this iterator.
  /// \param[in]  dim     The dimension in which to offset.
  /// \tparam     Value   The value which defines the dimension.
  fluidity_host_device constexpr self_t offset(int amount) const
  {
    return self_t{_ptr + amount * stride(dim_x), *this};
  }

  /// Offsets the iterator by \p amount in the dimension defined by \p dim, and
  /// returns a new iterator to the offset element.
  /// \param[in]  amount  The amount of offset from this iterator.
  /// \param[in]  dim     The dimension in which to offset.
  /// \tparam     Value   The value which defines the dimension.
/*
  template <std::size_t Value>
  fluidity_host_device constexpr self_t
  offset(int amount, Dimension<Value> dim) const
  {
    return self_t{_ptr + amount * stride(Dimension<Value>{}), *this};
  }
*/
  template <typename Dim>
  fluidity_host_device constexpr self_t offset(int amount, Dim dim) const
  {
    return self_t{_ptr + amount * stride(dim), *this};
  }

  /// Shifts the iterator by \p amount in dimension \p dim, modifying this
  /// iterator. This can shift the iterator forward and backward in the given
  /// dimension based on the sign of \p amount (+ = forward).
  /// \param[in]  amount  The amount to advance the iterator by.
  /// \param[in]  dim     The dimension to advance the iterator in.
  /// \tparam     Value   The value which defines the dimension.
  fluidity_host_device constexpr self_t& shift(int amount)
  {
    _ptr += amount * stride(dim_x);
    return *this;
  }

  /// Shifts the iterator by \p amount in dimension \p dim, modifying this
  /// iterator. This can shift the iterator forward and backward in the given
  /// dimension based on the sign of \p amount (+ = forward).
  /// \param[in]  amount  The amount to advance the iterator by.
  /// \param[in]  dim     The dimension to advance the iterator in.
  /// \tparam     Value   The value which defines the dimension.
/*
  template <std::size_t Value>
  fluidity_host_device constexpr self_t&
  shift(int amount, Dimension<Value> dim)
  {
    _ptr += amount * stride(Dimension<Value>{});
    return *this;
  }
*/
  template <typename Dim>
  fluidity_host_device constexpr self_t& shift(int amount, Dim dim)
  {
    _ptr += amount * stride(dim);
    return *this;
  }
  /// Returns the size of a given dimension of iteration. This is the number of
  /// elements which can be iterated over in the dimension.
  /// \param[in]  dim     The dimension to advance the iterator in.
  /// \tparam     Value   The value which defines the dimension.
/*
  template <std::size_t Value>
  fluidity_host_device constexpr std::size_t
  size(Dimension<Value> dim) const
  {
    return dim_info_t::size(Dimension<Value>());
  }
*/
  template <typename Dim>
  fluidity_host_device constexpr std::size_t size(Dim dim) const
  {
    return dim_info_t::size(dim);
  }
  /// Returns stride required to iterate in the dimension \p dim. The stride in
  /// the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <typename Dim> 
  fluidity_host_device constexpr std::size_t stride(Dim dim) const
  {
    return dim_stride(static_cast<const dim_info_t&>(*this), dim);
//    return dim_info_t::offset(Dimension<Value>{});
  }
};

#if defined(__CUDACC__)

/// Makes a multidimensional iterator over a multidimensional space, where the
/// properties of the space are defined by the DimInfo parameter. The DimInfo
/// template parameter must be of DimInfoCt type, otherwise a compiler error is
/// generated.
/// \tparam T       The type of the data to iterate over.
/// \tparam DimInfo The information which defines the multi dimensional space.
/// \tparam Padding The amount of padding for the iterator space.
template <typename T, typename DimInfo, std::size_t Padding = 0>
fluidity_device_only auto make_multidim_iterator()
{
  using iter_t = MultidimIterator<T, DimInfo, exec::gpu_type>;
  constexpr std::size_t buffer_size =
    DimInfo().template total_size<Padding>();

  __shared__ T buffer[buffer_size];
  return iter_t{buffer};
}

/// Makes a multidimensional iterator over a multidimensional space, where the
/// properties of the space are defined by the DimInfo parameter. The DimInfo
/// template parameter must be of DimInfoCt type, otherwise a compiler error is
/// generated.
/// \tparam T       The type of the data to iterate over.
/// \tparam DimInfo The information which defines the multi dimensional space.
/// \tparam PadInfo The information for the padding.
template <typename T, typename DimInfo, typename PadInfo>
fluidity_device_only auto make_multidim_iterator()
{
  using iter_t = MultidimIterator<T, DimInfo, exec::gpu_type>;
  constexpr std::size_t buffer_size =
    DimInfo().template total_size<PadInfo>();

  __shared__ T buffer[buffer_size];
  return iter_t{buffer};
}

/// Makes a multidimensional iterator over a multidimensional space, where the
/// properties of the space are defined by the DimInfo parameter. The DimInfo
/// template parameter must be of DimInfoCt type, otherwise a compiler error is
/// generated.
/// \param[in] ptr     A pointer to the start of data to iterate over.      
/// \tparam    T       The type of the data to iterate over.
/// \tparam    DimInfo The information which defines the multi dimensional
///                    space.
template <typename T, typename DimInfo>
fluidity_device_only constexpr auto make_multidim_iterator(T* ptr)
{
  using iter_t = MultidimIterator<T, DimInfo, exec::gpu_type>;
  return iter_t{ptr};
}

/// Makes a multidimensional iterator over a multidimensional space, where the
/// properties of the space are defined by the DimInfo parameter. The DimInfo
/// template parameter must be of DimInfoCt type, otherwise a compiler error is
/// generated.
/// \param[in] ptr     A pointer to the start of data to iterate over.      
/// \tparam    T       The type of the data to iterate over.
/// \tparam    DimInfo The information which defines the multi dimensional space.
template <typename T, typename DimInfo>
fluidity_device_only constexpr auto make_multidim_iterator(T* ptr, DimInfo info)
{
  using iter_t = MultidimIterator<T, DimInfo, exec::gpu_type>;
  return iter_t{ptr, info};
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
  static_assert(fluid::is_same_v<DimInfo, DimInfoCt>,
                "DimInfo must be DimInfoCt type to make a multidim iterator!");
  static thread_local T buffer[DimInfo::total_size()];
  return iter_t{buffer};
}

#endif // __CUDACC__

} // namespace fluid

#endif // FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_HPP