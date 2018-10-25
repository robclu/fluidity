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

#include "setting.hpp"
#include <fluidity/utility/string.hpp>
//#include <cstddef>
//#include <iomanip>
//#include <iostream>
#include <fstream>
#include <streambuf>

namespace fluid   {
namespace setting {

/// The Settings class holds a list of settings.
class Settings {
  //using setting_t            = std::unique_ptr<Setting>;
  /// Defines the type of the container for the settings.
  //using settings_container_t = std::vector<Setting>;

  /// Defines the type of the delimeter used for the settings.
  static constexpr const char delim   = ':';
  /// Defines the type of a comment.
  static constexpr const char comment = '#';
  /// Defines a space character.
  static constexpr const char space   = ' ';
  /// Defines the separator for setting types.
  static constexpr const char sep     = ',';
  /// Defines the identifier for opening an object.
  static constexpr const char open    = '{';
  /// Defines the identifier for closing an object.
  static constexpr const char close   = '}'; 

 public:
  /// Returns a list of settings from a string \p data.
  /// \param[in] data The data to load the settings from.
  static auto from_string(const std::string& data)
  {
    std::vector<Setting> settings;

    std::size_t start = 0, match = 0; 
    int setting_end = 0;
    do {
      // Know that the form is <name> : <value>, so look for the :
      if ((match = data.find(delim, start)) == std::string::npos)
        break;

      settings.emplace_back();
      auto& setting = settings.back();

      setting.name = data.substr(start, match - start);
      util::remove(setting.name, ' ', '{', '\n');
      
      start = ++match;
      while (setting_end != -1)
      {
        auto& c = data[match++];
        if (c == open)
        {
          setting_end += 1;
          setting.complex = true;
        }
        if ((c == sep && !setting_end) || c == close)
        {
          setting_end -= 1;
        }
      }
      setting.value = data.substr(start, match - start);
      start         = ++match;
      setting_end   = 0;
      //std::cout << setting << "\n";
    } while (match != std::string::npos);
    return settings;
  }

  /// Returns a list of settings from a file at the location \path.
  /// \param[in] path The path to the file to load the settings from.
  static auto from_file(const std::string& path)
  {
    std::string   data;
    std::ifstream f(path);

    f.seekg(0, std::ios::end);
    data.reserve(f.tellg());
    f.seekg(0, std::ios::beg);
    data.assign((std::istreambuf_iterator<char>(f)),
                 std::istreambuf_iterator<char>());
    return from_string(data);
  }

 //private:

/*
  /// Loads the settings container with the settings in the file at the \p path.
  /// \param[in] path The path to the settings file.
  void load_from_file(const std::string& path)
  {    std::string   line, data;
    std::ifstream f(path);

    f.seekg(0, std::ios::end);
    data.reserve(f.tellg());
    f.seekg(0, std::ios::beg);
    data.assign((std::istreambuf_iterator<char>(f)),
                 std::istreambuf_iterator<char>());

    std::size_t pos = 0, res = 0; 
    int it = 0, term = 0;
    do {
      if ((res = data.find(delim, pos)) == std::string::npos)
        break;

      _settings.emplace_back();
      auto& setting = _settings.back();

      setting.name = data.substr(pos, res - pos);
      util::remove(setting.name, ' ', '{', '\n');
      
      pos = ++res;
      while (term != -1)
      {
        auto& c = data[res++];
        if (c == open)
        {
          term += 1;
          setting.complex = true;
        }
        if ((c == sep && !term) || c == close)
        {
          term -= 1;
        }
      }
      term = 0;
      setting.value = data.substr(pos, res - pos);
      //util::remove(setting.value, ' ', '\n');
      pos = ++res;
      std::cout << setting << "\n";
    } while (res != std::string::npos);

  }
*/
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_CONFIGURE_HPP