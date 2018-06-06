//==--- fluidity/iterator/iterator_traits.hpp -------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  iterator_traits.hpp
/// \brief This file defines traits for iterators.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ITERATOR_ITERATOR_TRAITS_HPP
#define FLUIDITY_ITERATOR_ITERATOR_TRAITS_HPP

#include <type_traits>

namespace fluid {

template <typename Iterator>
using iter_value_t = std::decay_t<Iterator>::value_t;

} // namespace fluid

#endif // FLUIDITY_ITERATOR_ITERATOR_TRAITS_HPP