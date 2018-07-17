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
  add  = 0,   //!< Adds elements while folding.
  sub  = 1,   //!< Subtracts elements while folding.
  mult = 2,   //!< Multiplies elements while folding.
  div  = 3    //!< Divides elements while folding.
};

namespace detail {

/// Defines a struct which implements folding a set of compile time values using
/// a specific operation.
/// \tparam Op     The operation to use while folding.
/// \tparam Values The values to fold.
template <FoldOp Op, int... Values> struct Folder;

/// Specialization of the folding struct for the multiplication operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <int First, int... Rest>
struct Folder<FoldOp::mult, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::mult, Rest...>;
    return First * fold_t::apply();
  }
};

/// Specialization of the folding struct for the multiplication operation.
template <>
struct Folder<FoldOp::mult> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    return 1;
  }
};

/// Specialization of the folding struct for the division operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <int First, int... Rest>
struct Folder<FoldOp::div, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::div, Rest...>;
    return First / fold_t::apply();
  }
};

/// Specialization of the folding struct for the division operation.
template <>
struct Folder<FoldOp::div> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    return 1;
  }
};

/// Specialization of the folding struct for the addition operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <int First, int... Rest>
struct Folder<FoldOp::add, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::add, Rest...>;
    return First + fold_t::apply();
  }
};

/// Specialization of the folding struct for the addition operation.
template <>
struct Folder<FoldOp::add> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    return 0;
  }
};

/// Specialization of the folding struct for the subtraction operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <int First, int... Rest>
struct Folder<FoldOp::sub, First, Rest...> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    using fold_t = Folder<FoldOp::sub, Rest...>;
    return First - fold_t::apply();
  }
};

/// Specialization of the folding struct for the subtraction operation.
/// \tparam First The first value to fold.
/// \tparam Rest  The rest of the values to fold.
template <>
struct Folder<FoldOp::sub> {
  /// Applies the folding, returning the result of folding the compile time list
  /// of values.
  static constexpr decltype(auto) apply()
  {
    return 0;
  }
};

} // namespace detail

/// Folds (left fold) the Values using the defined Op.
/// \tparam Op     The operation to use while folding.
/// \tparam Values The list of values to fold.
template <FoldOp Op, int... Values>
fluidity_host_device constexpr auto fold()
{
  return detail::Folder<Op, Values...>::apply();
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_FOLD_HPP