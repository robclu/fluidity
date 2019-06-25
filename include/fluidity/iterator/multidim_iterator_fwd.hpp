//==--- fluidity/iterator/multidim_iterator_fwd.hpp -------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multidim_iterator_fwd.hpp
/// \brief This file forward declares a multi dimensional iterator.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_FWD_HPP
#define FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_FWD_HPP

#include <fluidity/dimension/dimension.hpp>
#include <fluidity/execution/execution_policy.hpp>

namespace fluid {

/// Forward declaration of a class for multi-dimensional iterator.
/// \tparam T         The type of the data to iterate over.
/// \tparam DimInfo   Information for the dimensions.
/// \tparam Exec      The execution policy for the iterator.
template <typename T,
          typename DimensionInfo = DimInfo<2>,
          typename Exec          = exec::default_exec_t>
struct MultidimIterator;

} // namespace fluid

#endif // FLUIDITY_ITERATOR_MULTIDIM_ITERATOR_FWD_HPP
