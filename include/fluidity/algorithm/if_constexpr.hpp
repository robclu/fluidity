//==--- fluidity/algorithm/if_constexpr.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  if_constexpr.hpp
/// \brief This file implements a wrapper which performs a compile time if
///        expression as does if constexpr in c++ 17. Since cuda does not yet
///        support c++17 we need a wrapper in the meantime.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_IF_CONSTEXPR_HPP
#define FLUIDITY_ALGORITHM_IF_CONSTEXPR_HPP

#include <fluidity/utility/portability.hpp>
#include <utility>

namespace fluid  {
namespace detail {

#if !defined(__CUDA_ARCH__)

/// Default implementation of the IfConstexpr class which operloads operator()
/// to compile time select between two predicates based on a compile time
/// conditional.
/// 
/// \note This can have multiple true conditions, and therefore is slower to
///       compile since we don't terminate.
///       
/// \tparam Conditions A compile time list of conditions where the nth condition
///                    determines if the nth predicate passed to the constructor
///                    must be executed.
template <bool... Conditions>
struct IfConstexpr {
  /// Does nothing, this is the termination case.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename... Ps>
  constexpr IfConstexpr(Ps&&...) {}

  /// Invokes the predicate \p p, as this is the case where there are no
  /// conditions but there is a predicate remaining, which is the case when
  /// only 
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename P>
  constexpr IfConstexpr(P&& p) { p(); }
};

/// Specialization for the case that a condition is true which invokes the 
/// predicate assosciated with the condition.
/// \tparam Conditions The remaining conditions to check.
template <bool...Conditions>
struct IfConstexpr<true, Conditions...> : public IfConstexpr<Conditions...> {
  /// Invokes the \p predicate since this is the true case.
  /// \param[in] p  The tredicate assosciated with the true condition.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    P  The type of the predicate.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename P, typename... Ps>
  constexpr IfConstexpr(P&& p, Ps&&... ps)
  : IfConstexpr<Conditions...>(std::forward<Ps>(ps)...)
  {
    p();
  }
};

/// Specialization for the case that a condition is false which does nothing
/// but forwards the conditions and predicates down the chain.
/// \tparam Conditions The remaining conditions to check.
template <bool...Conditions>
struct IfConstexpr<false, Conditions...> : public IfConstexpr<Conditions...> {
  /// Does nothing since this is the false case.
  /// \param[in] p  The tredicate assosciated with the true condition.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    P  The type of the predicate.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename P, typename... Ps>
  constexpr IfConstexpr(P&& /*p*/, Ps&&... ps)
  : IfConstexpr<Conditions...>(std::forward<Ps>(ps)...) {}

  /// Does nothing since this is the false case.
  /// \param[in] p  The tredicate assosciated with the true condition.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    P  The type of the predicate.
  /// \tparam    Ps The types of the rest of the predicates.
  constexpr IfConstexpr() {}
};

} // namespace detaild

/// Invokes one of the predicates if the corresponding compile time condtion
/// is true, otherwise does nothing. Supplying only a single condition will
/// cause early terminations and result in faster compile times.
/// \param[in] p          The first predicate for the Condition.
/// \param[in] ps         The rest of the predicates.
/// \tparam    Condition  The conditions for the \p p predicates.
/// \tparam    Conditions The conditions for the \p ps predicates.
/// \tparam    P          The type of the \p p predicate.
/// \tparam    Ps         The types of the \p ps predicates.
template <bool Condition, bool... Conditions, typename P, typename... Ps>
inline constexpr void if_constexpr(P&& p, Ps&&... ps)
{
  detail::IfConstexpr<Condition, !Condition>(
    std::forward<P>(p), std::forward<Ps>(ps)...
  );
}

#else

/// Default implementation of the IfConstexpr class which operloads operator()
/// to compile time select between two predicates based on a compile time
/// conditional.
/// 
/// \note This can have multiple true conditions, and therefore is slower to
///       compile since we don't terminate.
///       
/// \tparam Conditions A compile time list of conditions where the nth condition
///                    determines if the nth predicate passed to the constructor
///                    must be executed.
template <bool... Conditions>
struct IfConstexpr {
  /// Does nothing, this is the termination case.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename... Ps>
  fluidity_device_only IfConstexpr(Ps&&...) {}

  /// Invokes the predicate \p p, as this is the case where there are no
  /// conditions but there is a predicate remaining, which is the case when
  /// only 
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename P>
  fluidity_device_only IfConstexpr(P&& p) { p(); }
};

/// Specialization for the case that a condition is true which invokes the 
/// predicate assosciated with the condition.
/// \tparam Conditions The remaining conditions to check.
template <bool...Conditions>
struct IfConstexpr<true, Conditions...> : public IfConstexpr<Conditions...> {
  /// Invokes the \p predicate since this is the true case.
  /// \param[in] p  The tredicate assosciated with the true condition.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    P  The type of the predicate.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename P, typename... Ps>
  fluidity_device_only IfConstexpr(P&& p, Ps&&... ps)
  : IfConstexpr<Conditions...>(std::forward<Ps>(ps)...)
  {
    p();
  }
};

/// Specialization for the case that a condition is false which does nothing
/// but forwards the conditions and predicates down the chain.
/// \tparam Conditions The remaining conditions to check.
template <bool...Conditions>
struct IfConstexpr<false, Conditions...> : public IfConstexpr<Conditions...> {
  /// Does nothing since this is the false case.
  /// \param[in] p  The tredicate assosciated with the true condition.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    P  The type of the predicate.
  /// \tparam    Ps The types of the rest of the predicates.
  template <typename P, typename... Ps>
  fluidity_device_only IfConstexpr(P&& /*p*/, Ps&&... ps)
  : IfConstexpr<Conditions...>(std::forward<Ps>(ps)...) {}

  /// Does nothing since this is the false case.
  /// \param[in] p  The tredicate assosciated with the true condition.
  /// \param[in] ps The predicates assosciated with the other conditions.
  /// \tparam    P  The type of the predicate.
  /// \tparam    Ps The types of the rest of the predicates.
  fluidity_device_only IfConstexpr() {}
};

} // namespace detaild

/// Invokes one of the predicates if the corresponding compile time condtion
/// is true, otherwise does nothing. Supplying only a single condition will
/// cause early terminations and result in faster compile times.
/// \param[in] p          The first predicate for the Condition.
/// \param[in] ps         The rest of the predicates.
/// \tparam    Condition  The conditions for the \p p predicates.
/// \tparam    Conditions The conditions for the \p ps predicates.
/// \tparam    P          The type of the \p p predicate.
/// \tparam    Ps         The types of the \p ps predicates.
template <bool Condition, bool... Conditions, typename P, typename... Ps>
fluidity_device_only inline void if_constexpr(P&& p, Ps&&... ps)
{
  detail::IfConstexpr<Condition, !Condition>(
    std::forward<P>(p), std::forward<Ps>(ps)...
  );
}

#endif // __CUDA_ARCH__

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_IF_CONSTEXPR_HPP
