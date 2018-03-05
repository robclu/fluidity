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
#include <vector>

namespace fluid {

/// The DimInfoCt struct defines dimension information which is known at compile
/// time, where the dimension sizes are built into the type via the template
/// parameters. All functions are also compile-time computed.
/// \tparam Sizes The sizes of the dimensions.
template <std::size_t... Sizes>
struct DimInfoCt {
  /// Returns the number of dimensions in the space.
  fluidity_host_device static constexpr auto num_dimensions()
  {
    return sizeof...(Sizes);
  }

  /// Returns the size of the \p nth dimension.
  /// \param[in] n The index of the dimension to get the size of.
  fluidity_host_device static constexpr std::size_t size(std::size_t i)
  {
    constexpr std::size_t sizes[sizeof...(Sizes)] = { Sizes... };
    return sizes[i];
  }

  /// Returns the size of the \p dim dimension.
  /// \param[in] dim    The dimension to get the size of.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device static constexpr std::size_t
  size(Dimension<Value> /*dim*/)
  {
      constexpr std::size_t sizes[sizeof...(Sizes)] = { Sizes... };
      return sizes[Value];
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension 
  fluidity_host_device static constexpr auto total_size()
  {
    return (1 * ... * Sizes);
  }

 private:
  /// Hides the implementation of the stride computation.
  struct detail {
    /// Computes the offset for a dimension \p dim, using a starting value of
    /// \p start_value.
    /// \param[in] dim          The dimension to compute the offset for.
    /// \param[in] start_value  The starting value for the computation.
    /// \tparam    Value        The value which defines the dimension.
    template <std::size_t Value>
    fluidity_host_device static constexpr std::size_t
    offset(Dimension<Value> /*dim*/, std::size_t start_value)
    {
      return start_value * size(Dimension<Value - 1>{});
    }
  };

 public:
  /// Returns amount of offset required to iterate in dimension \p dim. The
  /// offset in the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value> 
  fluidity_host_device static constexpr std::size_t
  offset(Dimension<Value> /*dim*/)
  {
    if constexpr (Value == 0)
    {
      return 1;
    }
    else
    {
      return detail::offset(Dimension<Value>{}, offset(Dimension<Value - 1>{}));
    }
  }

  /// Returns the index of an element in dimension Dim if \p index is the index
  /// as if the data was flattened (i.e linear).
  /// \param[in] index The index to get in a given dimension.
  /// \tparam    Value The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device static constexpr std::size_t
  flattened_index(std::size_t index, Dimension<Value> /*dim*/)
  {
    return index / offset(Dimension<Value>{}) % size(Value);
  }
};

/// The DimInfo struct defines dimension information which is known not known
/// at compile time.
struct DimInfo {
  fluidity_host_device DimInfo() = default;

  /// Sets the sizes of the dimensions.
  /// \param[in] sizes  The sizes of the dimensions. 
  /// \tparam    Sizes  The type of the sizes.
  template <typename... Sizes>
  fluidity_host_device DimInfo(Sizes&&... sizes)
  : _sizes{static_cast<std::size_t>(sizes)...} {}

  /// Returns the number of dimensions in the space.
  fluidity_host_device std::size_t num_dimensions() const
  {
    return _sizes.size();
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
      return _sizes[Value];
  }

  /// Returns the total size of the N dimensional space i.e the total number of
  /// elements in the space. This is the product sum of the dimension 
  fluidity_host_device std::size_t total_size() const
  {
    auto prod_sum = 1;
    for (const auto& value : _sizes)
    {
      prod_sum *= value;
    }
    return prod_sum;
  }

  /// Adds a dimension to the dimension information.
  /// \param[in] sz The size of the new dimension to add.
  fluidity_host_device void push_back(std::size_t sz)
  {
    _sizes.push_back(sz);
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

 public:
  /// Returns amount of offset required to iterate in dimension \p dim. The
  /// offset in the 0 dimension (Value = 0) is always taken to be one.
  /// \param[in] dim    The dimension to get the offset for.
  /// \tparam    Value  The value which defines the dimension.
  template <std::size_t Value> 
  fluidity_host_device std::size_t offset(Dimension<Value> /*dim*/) const
  {
    if constexpr (Value == 0)
    {
      return 1;
    }
    else
    {
      return detail::offset(Dimension<Value>{}            ,
                            offset(Dimension<Value - 1>{}),
                            *this                         );
    }
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
  std::vector<std::size_t> _sizes = {}; //!< Sizes of the dimensions.
};

} // namespace fluid

#endif // FLUIDITY_DIMENSION_DIMENSION_INFO_CT_HPP