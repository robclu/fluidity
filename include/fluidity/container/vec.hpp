//==--- fluidity/container/vec.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  vec.hpp
/// \brief This file defines a vector classes.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_VEC_HPP
#define FLUIDITY_CONTAINER_VEC_HPP

#include "array.hpp"

namespace fluid {

/// Alias for a 2 dimensional vector of type T.
/// \tparam T The type of the vector data.
template <typename T>
using Vec2 = Array<T, 2>;

/// Alias for a 3 dimensional vector of type T.
/// \tparam T The type of the vector data.
template <typename T>
using Vec3 = Array<T, 3>;

} // namespace fluid

#endif // FLUIDITY_CONTAINER_VEC_HPP

