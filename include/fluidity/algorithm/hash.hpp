//==--- fluidity/algorithm/hash.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  hash.cu
/// \brief This file defines a constexpr implementation of a hash function.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_HASH_HPP 
#define FLUIDITY_ALGORITHM_HASH_HPP 

#include <fluidity/utility/portability.hpp>

namespace fluid {

/// This computes a hash of the \p input, returning the result.
/// \param[in] input The input to compute the hash of.
fluidity_host_device constexpr auto hash(char const* input) -> unsigned int {
  return *input 
    ? static_cast<unsigned int>(*input) + 33 * hash(input + 1)
    : 5381;
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_HASH_HPP 
