//==--- fluidity/algorithm/fold.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  accumulate.hpp
/// \brief This file defines the implementation of folding with a specific
///        operation. This is essentially a fold operation, but cuda does not
///        have c++17 support so we can't use the built in fold operations.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_FOLD_HPP
#define FLUIDITY_ALGORITHM_FOLD_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {

/// Defines the type of operation for the folding.
enum class FoldOp {
  Add  = 0,   //!< Adds elements while folding.
  Sub  = 1,   //!< Subtracts elements while folding.
  Mult = 2,   //!< Multiplies elements while folding.
  Div  = 3    //!< Divides elements while folding.
};

namespace detail {

/// Defines a struct which implements folding a set of compile time values using
/// a specific operation.
/// \tparam Op     The operation to use while folding.
/// \tparam Values The values to fold.
template <FoldOp Op, typename... Values> struct Folder;

/// Specialization of the folding struct for the multiplication operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <typename First, typename... Rest>
struct Folder<FoldOp::Mult, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  fluidity_host_device static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::Mult, Rest...>;
    return First{} * (sizeof...(Rest) > 0 ? fold_t::apply() : 1);
  }
};

/// Specialization of the folding struct for the division operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <typename First, typename... Rest>
struct Folder<FoldOp::Div, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  fluidity_host_device static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::Div, Rest...>;
    return First{} * (sizeof...(Rest) > 0 ? fold_t::apply() : 1);
  }
};

/// Specialization of the folding struct for the addition operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <typename First, typename... Rest>
struct Folder<FoldOp::Add, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  fluidity_host_device static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::Add, Rest...>;
    return First{} + (sizeof...(Rest) > 0 ? fold_t::apply() : 0);
  }
};

/// Specialization of the folding struct for the subtraction operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <typename First, typename... Rest>
struct Folder<FoldOp::Sub, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  fluidity_host_device static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::Sub, Rest...>;
    return First{} + (sizeof...(Rest) > 0 ? fold_t::apply() : 0);
  }
};

} // namespace detail

/// Folds (left fold) the Values using the defined Op.
/// \tparam Op     The operation to use while folding.
/// \tparam Values The list of values to fold.
template <FoldOp Op, typename... Values>
fluidity_host_device decltype(auto) accumulate()
{
  return (sizeof...(Values) > 0)
         ? detail::Folder<Op, Values...>::apply() 
         : (Op == FoldOp::Add || Op == FoldOp::Sub) ? 0 : 1;
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_FOLD_HPP