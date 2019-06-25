//==--- fluidity/setting/option_holder.hpp ----------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  option_holder.hpp
/// \brief This file defines a utility do define a type option and an associated
///        string which can be used to set the value of the option.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_OPTION_HOLDER_HPP
#define FLUIDITY_SETTING_OPTION_HOLDER_HPP

namespace fluid   {
namespace setting {

/// The OptionHolder class holds and option of type T and an assosciated string
/// which defines the value which can be used to set the option at runtime.
/// \tparam T The type of the option.
template <typename T>
struct OptionHolder {
  /// Defines the type of the option.
  using type = T;
  
  /// Constructor to set the value associated with the option.
  constexpr OptionHolder(const char* const v) : value(v) {};
    
  const char* const value; //!< The value of the option being held.
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_OPTION_HOLDER_HPP