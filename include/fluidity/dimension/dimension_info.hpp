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

namespace fluid {

/// The DimInfoCt struct defines dimension information which is known at compile
/// time, where the dimension sizes are built into the type via the template
/// parameters. All functions are also compile-time computed.
/// \tparam Sizes The sizes of the dimensions.
template <std::size_t... Sizes>
struct DimInfoCt {
 public:
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
};

} // namespace fluid

#endif // FLUIDITY_DIMENSION_DIMENSION_INFO_CT_HPP