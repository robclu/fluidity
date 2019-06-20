//==--- fluidity/utility/contants.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  constants.hpp
/// \brief This file provides useful constants.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_CONSTANTS_HPP
#define FLUIDITY_UTILITY_CONSTANTS_HPP

namespace fluid {
namespace cx    {

/// Defines a constant value of zero.
static constexpr auto zero    = double{0};
/// Defines a constant value of a quarter.
static constexpr auto quarter = double{1} / double{4};
/// Defines  a constant value of half.
static constexpr auto half    = double{0.5};
/// Defines a constant value of a third.
static constexpr auto third   = double{1} / double{3};
/// Defines a constant value of one.
static constexpr auto one     = double{1};
/// Defines a constant value of two;
static constexpr auto two     = double{2};
/// Defines a constant value of $sqrt(2)$. Floating point numbers can store:
///   $'log_{10}(2^n)$
/// digits of precision, where $n$ is the number of bits used for the fractional
/// part of the number, So for $n=53$ for doubles, we can store approx 16
/// decimals. We are extra safe here, and set the number using a few more.
static constexpr auto root_2  = double{1.4142135623730950488016887242096980785};

}} // namespace fluid::cx

#endif // FLUIDITY_UTILITY_CONSTANTS_HPP
