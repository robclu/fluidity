//==--- fluidity/setting/setting.hpp ----------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  setting.hpp
/// \brief This file defines a Setting base class with general setting
///        which can then be extended for more specific settings.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_SETTING_HPP
#define FLUIDITY_SETTING_SETTING_HPP

#include <fluidity/utility/string.hpp>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <fstream>

namespace fluid   {
namespace setting {

/// The Setting class holds the name of a setting, and a value for the setting.
struct Setting {
  /// Constructor to create a setting from a string, where the name of the
  /// setting and the setting 
  /// \param[in] seq    The sequence to extact the setting from.
  /// \param[in] delim  The delimeter for the sequence.
  Setting() {}

  Setting(const std::string& seq, char delim)
  {
    auto results = util::tokenize(seq, delim, true);
    if (results.size() != 2)
    {
      // logging::logger_t::log(logging::debug_v, 
      //   "Attempt to create a setting which does not have the form:\n\n't"
      //   "setting_name delimeter setting_value\n");
    }
    else
    {
      name  = results.front();
      value = results.back();
    }
  }

  /// Returns true if the setting is empty.
  bool is_empty() const { return name.size() == 0 || value.size() == 0; }

  /// Overload of stream operator to output a setting to a stream.
  /// \param[in] o The output stream to output to.
  /// \param[in] s The setting to output to the stream.
  friend std::ostream& operator<<(std::ostream& o, const Setting& s)
  {
    o << "{\n    setting-name  : " 
      << std::right    << std::setw(15) << s.name
      << "\n    setting-type  : " 
      << std::right    << std::setw(15) << (s.complex ? "complex" : "simple")
      << "\n    setting-value : " 
      << std::right    << std::setw(15) << s.value
      << "\n}";
    return o;
  }

  std::string name    = "";     //!< The name of the setting.
  std::string value   = "";     //!< The value of the setting.
  bool        complex = false;  //!< If the setting is complex (value is not  
                                //!< just a value).
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_SETTING_HPP