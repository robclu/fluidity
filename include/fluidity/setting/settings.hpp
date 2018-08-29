//==--- fluidity/setting/configure.hpp --------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  settings.hpp
/// \brief This file defines functionality to load a list of settings.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_CONFIGURE_HPP
#define FLUIDITY_SETTING_CONFIGURE_HPP

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
      << "\n    setting-value : " 
      << std::right    << std::setw(15) << s.value
      << "\n}";
    return o;
  }

  std::string name  = "";     //!< The name of the setting.
  std::string value = "";     //!< The value of the setting.
  bool        used  = false;  //!< If the setting has been used.
};

/// The Settings class holds a list of settings.
class Settings {
  /// Defines the type of the container for the settings.
  using settings_container_t = std::vector<Setting>;

  /// Defines the type of the delimeter used for the settings.
  static constexpr const char delim   = ':';
  /// Defines the type of a comment.
  static constexpr const char comment = '#';
  /// Defines a space character.
  static constexpr const char space   = ' ';

 public:
  /// Constructor to create the settings from a file.
  /// \param[in] path The path to the file to load the settings from.
  Settings(const std::string& path)
  {
    load_from_file(path);
  }

  /// Overload of operator[] to get the \p ith setting.
  /// \param[in] i The index of the setting to get.
  Setting& operator[](std::size_t i)
  {
    return _settings[i];
  }

  /// Returns the total number of settings.
  std::size_t size() const { return _settings.size(); }

  /// Returns a const iterator to the start of the settings.
  auto begin() { return _settings.begin(); }

  /// Returns a const iterator to the end of the settings.
  auto end() { return _settings.end(); }

 private:
  settings_container_t _settings; //!< The list of settings.

  /// Loads the settings container with the settings in the file at the \p path.
  /// \param[in] path The path to the settings file.
  void load_from_file(const std::string& path)
  {
    std::string   line;
    std::ifstream f(path);

    if (!f.is_open())
    {
      // logging::logger_t::log(logging::debug_v,
      //   "Failed to open the settings file: %s", path.c_str());
      return;
    }

    while (std::getline(f, line))
    {
      // Check to see that this is a valid line
      int start = 0;
      while (line[start] == space) { start++; }
      if (line[start] == comment)  { continue; }

      _settings.emplace_back(Setting(line, delim));
      if (_settings.back().is_empty())
      {
        _settings.pop_back();
      }
    }
    for (const auto& setting : _settings)
    {
      std::cout << setting << "\n";
    }
  }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_CONFIGURE_HPP