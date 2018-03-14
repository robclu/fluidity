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

namespace fluid  {
namespace detail {

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
struct IfConstexpr {};

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
    fluidity_host_device constexpr IfConstexpr(P&& p, Ps&&... ps)
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
    /// Invokes the \p predicate since this is the true case.
    /// \param[in] p  The tredicate assosciated with the true condition.
    /// \param[in] ps The predicates assosciated with the other conditions.
    /// \tparam    P  The type of the predicate.
    /// \tparam    Ps The types of the rest of the predicates.
    template <typename P, typename... Ps>
    fluidity_host_device constexpr IfConstexpr(P&& p, Ps&&... ps)
    : IfConstexpr<Conditions...>(std::forward<Ps>(ps)...) {}
};

} // namespace detail

/// Wrapper 
template <bool Condition, typename P1, typename P2>
fluidity_host_device inline constexpr decltype(auto)
if_constexpr(P1&& p1, P2&& p2)
{
  IfConstexpr<Condition>()(std::forward<P1>(p1), std::forward<P2>(p2));
}


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
fluidity_host_device inline constexpr void if_constexpr(P&& p, Ps&&... ps)
{
  detail::IfConstexpr
  <
    Condition, sizeof...(Conditions) == 0 ? !Condition : Conditions...
  >(std::forward<P>(p), std::forward<Ps>(ps)...);
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_IF_CONSTEXPR_HPP
