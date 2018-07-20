//==--- fluidity/dimension/dimension_info.hpp -------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dimension_info.hpp
/// \brief This file defines a struct which holds dimension information at
///        both compile time and at runtime.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_DIMENSION_DIMENSION_INFO_CT_HPP
#define FLUIDITY_DIMENSION_DIMENSION_INFO_CT_HPP

#include "dimension.hpp"
#include <fluidity/algorithm/if_constexpr.hpp>
#include <fluidity/algorithm/unrolled_for.hpp>
#include <fluidity/algorithm/fold.hpp>
#include <vector>

namespace fluid  {
namespace detail {

/// Dispatch tag struct for dispatching based on the number of dimensions.
template <std::size_t> struct DimDispatchTag {};

} // namespace detail

/// Defines the type of a 1D dispatch tag.
using dispatch_tag_1d_t = detail::DimDispatchTag<1>;
/// Defines the type of a 2D dispatch tag.
using dispatch_tag_2d_t = detail::DimDispatchTag<2>;
/// Defines the type of a 3D dispatch tag.
using dispatch_tag_3d_t = detail::DimDispatchTag<3>;

/// Creates a constexpr instance of a dimension dispatch tag from a type which
/// contains a constexpr dimensions trait.
/// \tparam T The type which has dimension information.
//template <typename T>
//static constexpr auto dim_dispatch_tag = 
//  detail::DimDispatchTag<std::decay_t<T>::dimensions>{};

/// Creats a contexpr instance of a dimension dispatch tag from a number of
/// dimensions.
/// \tparam Dims The number of dimensions.
template <std::size_t Dims>
static constexpr auto dim_dispatch_tag = detail::DimDispatchTag<Dims>{};

/// Defines a struct for padding information for a specific dimension.
/// \tparam Dim     The dimension the padding applied to.
/// \tparam Amount  The amount of padding for the dimension.
template <std::size_t Dim, std::size_t Amount>
struct PaddingInfo {
  /// Defines the dimension the padding applies to.
  static constexpr auto dim    = Dim;
  /// Defines the amount of the padding for the dimension.
  static constexpr auto amount = Amount;
};

/// The DimInfoCt struct defines dimension information which is known at compile
/// time, where the dimension sizes are built into the type via the template
/// parameters. All functions are also compile-time computed.
/// \tparam Sizes The sizes of the dimensions.
template <std::size_t... Sizes>
struct DimInfoCt {
  /// Defines that the offset computations are constexpr.
  static constexpr auto constexpr_offsets = true;
  /// Defines the number of dimensions which can be used with the type.
  static constexpr auto dimensions        = sizeof...(Sizes);

  /// Returns the number of dimensions in the space.
  constexpr auto num_dimensions() const
  {
    return sizeof...(Sizes);
  }

  /// Returns the size of the \p nth dimension.
  /// \param[in] n The index of the dimension to get the size of.
  constexpr std::size_t size(std::size_t i) const
  {
    constexpr std::size_t sizes[sizeof...(Sizes)] = { Sizes... };
    return sizes[i];
  }

  /// Returns the size of the \p dim dimension.
  /// \param[in] dim    The dimension to get the size of.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value>
  constexpr std::size_t size(Dimension<Value> /*dim*/) const
  {
      constexpr std::size_t sizes[sizeof...(Sizes)] = { Sizes... };
      return sizes[Value];
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimensions.
  constexpr std::size_t total_size() const
  {
    return fold<FoldOp::mult, Sizes...>();
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space if the space is padded with Padding amount.
  /// \tparam Padding A padding value for each dimension.
  template <std::size_t Padding>
  constexpr std::size_t total_size() const
  {
    return fold<FoldOp::mult, (Sizes + (Padding << 1))...>();
  }

  /// Returns the total size of the dimensional space when the padding info for
  /// a specific dimension is defined by the PaddingInfo.
  /// \param[in] PaddingInfo The information which defines the padding.
  template <typename PaddingInfo>
  constexpr std::size_t total_size() const
  {
    return total_size_impl<PaddingInfo, 0, Sizes...>();
  }

 private:
  /// Defines the type of the dimension information.
  using dim_info_t = DimInfoCt<Sizes...>;

  /// Hides the implementation of the stride computation.
  struct detail {
    /// Computes the offset for a dimension \p dim, using a starting value of
    /// \p start_value.
    /// \param[in] start_value  The starting value for the computation.
    /// \tparam    Value        The value which defines the dimension.
    template <std::size_t Value>
    static constexpr std::size_t
    offset(Dimension<Value>, std::size_t start_value)
    {
      return start_value * dim_info_t().size(Dimension<Value - 1>{});
    }
  };

  /// Computes the total size of the dimensional space when additional sizes are
  /// defined for a dimension.
  /// \tparam Info The additional size information.
  /// \tparam I    The index of the iteration.
  /// \tparam S    The size of the dimension at iteration I.
  /// \tparam Szs  The sizes of the remaining dimensions.
  template <typename Info, std::size_t I, std::size_t S, std::size_t... Szs>
  constexpr std::size_t total_size_impl() const
  {
    constexpr auto zero = std::size_t{0};
    constexpr auto size = S + (Info::dim == I ? (Info::amount << 1) : zero);
    return size * total_size_impl<Info, I + 1, Szs...>();
  }

  /// Computes the total size of the dimensional space when additional sizes are
  /// defined for a dimension. This is the terminating case.
  /// \tparam Info The additional size information.
  /// \tparam I    The index of the iteration.
  template <typename Info, std::size_t I>
  constexpr std::size_t total_size_impl() const
  {
    return std::size_t{1};
  }

 public:
  /// Returns amount of offset required to iterate in dimension \p dim. The
  /// offset in the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value, std::enable_if_t<(Value > 0), int> = 0> 
  constexpr std::size_t offset(Dimension<Value> /*dim*/) const
  {
    return detail::offset(Dimension<Value>{}            ,
                          offset(Dimension<Value - 1>{}));
  }

  /// Returns amount of offset required to iterate in dimension \p dim. The
  /// offset in the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value, std::enable_if_t<(Value <= 0), int> = 0> 
  constexpr std::size_t offset(Dimension<Value> /*dim*/) const
  {
    return 1;
  }

  /// Returns the index of an element in dimension Dim if \p index is the index
  /// as if the data was flattened (i.e linear).
  /// \param[in] index The index to get in a given dimension.
  /// \tparam    Value The value which defines the dimension.
  template <std::size_t Value>
  constexpr std::size_t
  flattened_index(std::size_t index, Dimension<Value> /*dim*/) const
  {
    return index / offset(Dimension<Value>{}) % size(Value);
  }
};

/// The DimInfo struct defines dimension information where the sizes of the
/// dimensions are not known at compile time, but the number of dimensions is.
/// \tparam Dims The number of dimensions for which information is provided.
template <std::size_t Dims>
struct DimInfo {
  /// Defines that the offset computations are constexpr.
  static constexpr auto constexpr_offsets = false;
  /// Defines the number of dimensions, which can be used with the type.
  static constexpr auto dimensions        = Dims;

  /// Default constructor -- enables creation of empty dimension information.
  DimInfo() = default;

  /// Sets the sizes of the dimensions.
  /// \param[in] sizes  The sizes of the dimensions. 
  /// \tparam    Sizes  The type of the sizes.
  template <typename... Sizes>
  fluidity_host_device DimInfo(Sizes&&... sizes)
  : _sizes{static_cast<std::size_t>(sizes)...}
  {
    static_assert(sizeof...(Sizes) == Dims,
                  "Size for each dimension is not provided!");
  }

  /// Returns the number of dimensions in the space.
  constexpr auto num_dimensions() const
  {
    return Dims;
  }

  /// Returns the size of the \p nth dimension.
  /// \param[in] n The index of the dimension to get the size of.
  fluidity_host_device std::size_t size(std::size_t i) const
  {
    return _sizes[i];
  }

  /// Returns the size of the \p dim dimension.
  /// \param[in] dim    The dimension to get the size of.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device std::size_t size(Dimension<Value> /*dim*/) const
  {
    static_assert(Value < Dims,
                  "Accessing the size of a dimension which does not exist!");
    return _sizes[Value];
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension 
  fluidity_host_device std::size_t total_size() const
  {
    std::size_t prod_sum = 1;
    for (const auto i : range(num_dimensions()))
    {
      prod_sum *= _sizes[i];
    }
    return prod_sum;
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space with twice the amount of Padding in each dimension
  /// ... i.e when the space is uniformly padded. This is the product sum of the
  /// dimensions with the padding added.
  /// \tparam Padding The amount of padding for one side of each dimension.
  template <std::size_t Padding>
  fluidity_host_device std::size_t total_size() const
  {
    std::size_t prod_sum = 1;
    for (const auto i : range(num_dimensions()))
    {
      prod_sum *= _sizes[i] + (Padding << 1);
    } 
    return prod_sum;
  }

  /// Returns the total size of the dimensional space when the padding info for
  /// a specific dimension is defined by the PaddingInfo.
  /// \param[in] PaddingInfo The information which defines the padding.
  template <typename PaddingInfo>
  constexpr std::size_t total_size() const
  {
    std::size_t prod_sum = 1;
    for (const auto i : range(num_dimensions()))
    {
      prod_sum *= _sizes[i] 
                + (PaddingInfo::dim == i ? (PaddingInfo::amount << 1) : 0);
    } 
    return prod_sum;
  }

  /// Returns a reference to the size of dimension \p dim.
  /// \param[in] dim The dimension size to get a refernece to.
  fluidity_host_device std::size_t& operator[](std::size_t dim)
  {
    return _sizes[dim];
  }

 private:
  /// Hides the implementation of the stride computation.
  struct detail {
    /// Computes the offset for a dimension \p dim, using a starting value of
    /// \p start_value.
    /// \param[in] dim          The dimension to compute the offset for.
    /// \param[in] start_value  The starting value for the computation.
    /// \param[in] i            The dimension info to compute the offset for.
    /// \tparam    Value        The value which defines the dimension.
    template <std::size_t Value>
    fluidity_host_device static constexpr std::size_t
    offset(Dimension<Value> /*dim*/, std::size_t start_value, const DimInfo& i)
    {
      return start_value * i.size(Dimension<Value - 1>{});
    }
  };

  /// Computes the total size of the dimensional space when additional sizes are
  /// defined for a dimension.
  /// \tparam Info The additional size information.
  /// \tparam I    The index of the iteration.
  /// \tparam S    The size of the dimension at iteration I.
  /// \tparam Szs  The sizes of the remaining dimensions.
  template <typename Info, std::size_t I, std::size_t S, std::size_t... Szs>
  constexpr std::size_t total_size_impl() const
  {
    constexpr auto zero = std::size_t{0};
    constexpr auto size = S + (Info::dim == I ? (Info::amount << 1) : zero);
    return size * total_size_impl<Info, I + 1, Szs...>();
  }

  /// Computes the total size of the dimensional space when additional sizes are
  /// defined for a dimension. This is the terminating case.
  /// \tparam Info The additional size information.
  /// \tparam I    The index of the iteration.
  template <typename Info, std::size_t I>
  constexpr std::size_t total_size_impl() const
  {
    return std::size_t{1};
  }

 public:
  /// Returns amount of offset required to iterate in dimension \p dim. The
  /// offset in the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value, std::enable_if_t<(Value > 0), int> = 0> 
  fluidity_host_device std::size_t offset(Dimension<Value> /*dim*/) const
  {
    return detail::offset(Dimension<Value>{}            ,
                          offset(Dimension<Value - 1>{}),
                          *this                         );
  }

  /// Returns amount of offset required to iterate in dimension \p dim. The
  /// offset in the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value, std::enable_if_t<(Value <= 0), int> = 0> 
  fluidity_host_device std::size_t offset(Dimension<Value> /*dim*/) const
  {
    return 1;
  }

  /// Returns the index of an element in dimension Dim if \p index is the index
  /// as if the data was flattened (i.e linear).
  /// \param[in] index The index to get in a given dimension.
  /// \tparam    Value The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device std::size_t
  flattened_index(std::size_t index, Dimension<Value> /*dim*/) const
  {
    return index / offset(Dimension<Value>{}) % size(Value);
  }

 private:
  std::size_t _sizes[Dims] = {}; //!< Sizes of the dimensions.
};

} // namespace fluid

#endif // FLUIDITY_DIMENSION_DIMENSION_INFO_HPP