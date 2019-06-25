//==--- fluidity/setting/parameter/parameter_specifier.hpp - -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  parameter_specifier.hpp
/// \brief This file defines a wrapper class to define if a parameter is
///        required or optional.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_PARAMETER_SPECIFIER_HPP
#define FLUIDITY_SETTING_PARAMETER_PARAMETER_SPECIFIER_HPP

namespace fluid   {
namespace setting {

/// Defines a specifier for a parameter which specifies if the parameter is
/// required or not.
/// \tparam Param    The type of the parameter to specify.
/// \tparam Required If the paramter is required.
template <typename Param, bool Required>
struct ParamSpecifier : public Param {
  /// Defines the type of the parameter.
  using parameter_t = Param;

  /// Inherit all the constructors of the parameter.
  using Param::Param;

  /// Returns true if the parameter is required, otherwise returns false.
  constexpr auto is_required() const -> bool
  {
    return Required;
  }

  /// Returns a reference to the paramter being specifier.
  auto get_param() const -> const Param&
  {
    return static_cast<const Param&>(*this);
  }
};

/// Defines an alias for a required parameter.
/// \tparam P The type of the paramter which is required.
template <typename P>
using required_param_t = ParamSpecifier<P, true>;

/// Defines an alias for an optional paramter.
/// \tparam P The type of the paramter which is required.
template <typename P>
using optional_param_t = ParamSpecifier<P, false>;

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_PARAMETER_SPECIFIER_HPP